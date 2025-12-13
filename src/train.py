import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import TrajSeqDataset
from model import LSTMMultiHorizon

def ade_fde(pred, tgt):
    # pred, tgt: [B, K, 3] in meters (ENU)
    diff = pred - tgt
    dist = torch.linalg.norm(diff, dim=-1)  # [B, K]
    ade = dist.mean().item()
    fde = dist[:, -1].mean().item()
    return ade, fde

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--output_dir", type=str, default="outputs/run1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--normalize", action="store_true", default=False, 
                    help="Enable feature normalization (recommended)")
    ap.add_argument("--use_layer_norm", action="store_true", default=False,
                    help="Enable layer normalization in model (recommended)")
    ap.add_argument("--min_improvement", type=float, default=0.01,
                    help="Minimum improvement in validation ADE for early stopping")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create datasets with normalization
    ds_train = TrajSeqDataset(args.data_npz, split="train", normalize=args.normalize)
    train_stats = ds_train.get_stats()
    ds_val = TrajSeqDataset(args.data_npz, split="val", normalize=args.normalize, stats=train_stats)
    
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)

    pred_len = ds_train.Y.shape[1]
    input_size = ds_train.X.shape[2]
    model = LSTMMultiHorizon(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        pred_len=pred_len,
        use_layer_norm=args.use_layer_norm,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.SmoothL1Loss()

    best_val_ade = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_ade": [], "val_fde": [], "lr": []}

    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train samples: {len(ds_train)}, Val samples: {len(ds_val)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))
        
        # Step the scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        all_ade, all_fde = [], []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
                ade, fde = ade_fde(pred, yb)
                all_ade.append(ade)
                all_fde.append(fde)
        val_loss /= max(1, len(val_loader))
        val_ade = float(np.mean(all_ade))
        val_fde = float(np.mean(all_fde))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ade"].append(val_ade)
        history["val_fde"].append(val_fde)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_ADE={val_ade:.2f}m val_FDE={val_fde:.2f}m lr={current_lr:.6f}")

        # Early stopping with improved tolerance
        if val_ade < best_val_ade - args.min_improvement:
            best_val_ade = val_ade
            patience_counter = 0
            save_dir = os.path.join(args.output_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            # Save training stats for inference
            if train_stats is not None:
                np.savez(os.path.join(save_dir, "norm_stats.npz"), **train_stats)
            # Save model configuration
            import json
            model_config = {
                'hidden_size': args.hidden,
                'num_layers': args.layers,
                'dropout': args.dropout,
                'use_layer_norm': args.use_layer_norm
            }
            with open(os.path.join(save_dir, "config.json"), 'w') as f:
                json.dump(model_config, f, indent=2)
            with open(os.path.join(save_dir, "meta.txt"), "w") as f:
                f.write(f"epoch={epoch}\nval_ADE={val_ade}\nval_FDE={val_fde}\n")
                f.write(f"hidden_size={args.hidden}\nnum_layers={args.layers}\n")
                f.write(f"dropout={args.dropout}\nbatch_size={args.batch_size}\n")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))

    # Plot curves
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(history["train_loss"], label="train_loss")
    ax[0].plot(history["val_loss"], label="val_loss")
    ax[0].legend(); ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch")
    ax[1].plot(history["val_ade"], label="val_ADE (m)")
    ax[1].plot(history["val_fde"], label="val_FDE (m)")
    ax[1].legend(); ax[1].set_title("Val ADE/FDE"); ax[1].set_xlabel("Epoch")
    ax[2].plot(history["lr"], label="Learning Rate")
    ax[2].legend(); ax[2].set_title("Learning Rate Schedule"); ax[2].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=160)
    plt.close()
    
    print(f"\nTraining complete. Best val_ADE: {best_val_ade:.2f}m")

if __name__ == "__main__":
    main()