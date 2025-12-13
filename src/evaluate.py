import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

from dataset import TrajSeqDataset
from model import LSTMMultiHorizon

def ade_fde_np(pred, tgt):
    diff = pred - tgt
    dist = np.linalg.norm(diff, axis=-1)  # [B, K]
    ade = float(dist.mean())
    fde = float(dist[:, -1].mean())
    return ade, fde

def constant_velocity_baseline(X_hist, pred_len, dt=30):
    # X_hist: [H, F] with vx, vy, vz = features 0..2
    vx = X_hist[-5:, 0].mean()
    vy = X_hist[-5:, 1].mean()
    vz = X_hist[-5:, 2].mean()
    steps = np.arange(1, pred_len+1, dtype=np.float32).reshape(-1, 1)
    offsets = np.hstack([vx*steps*dt, vy*steps*dt, vz*steps*dt])
    return offsets  # [K, 3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--output_dir", type=str, default="outputs/run1")
    ap.add_argument("--normalize", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load normalization stats if available
    stats_path = os.path.join(args.model_dir, "norm_stats.npz")
    train_stats = None
    if os.path.exists(stats_path):
        stats_data = np.load(stats_path)
        train_stats = {
            'x_mean': stats_data['x_mean'],
            'x_std': stats_data['x_std'],
            'y_mean': stats_data['y_mean'],
            'y_std': stats_data['y_std']
        }
        print("Loaded normalization statistics from training.")

    # Load model config if available
    config_path = os.path.join(args.model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        print(f"Loaded model configuration: hidden={model_config['hidden_size']}, "
              f"layers={model_config['num_layers']}, dropout={model_config['dropout']}")
    else:
        # Default config if not saved
        print("Warning: config.json not found, using default model parameters")
        model_config = {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'use_layer_norm': False
        }

    ds_test = TrajSeqDataset(args.data_npz, split="test", normalize=args.normalize, stats=train_stats)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    pred_len = ds_test.Y.shape[1]
    input_size = ds_test.X.shape[2]
    model = LSTMMultiHorizon(
        input_size=input_size,
        hidden_size=model_config.get('hidden_size', 128),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.2),
        pred_len=pred_len,
        use_layer_norm=model_config.get('use_layer_norm', False)
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device))
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Test"):
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    
    # Denormalize if stats are available
    if train_stats is not None and args.normalize:
        pred = pred * train_stats['y_std'] + train_stats['y_mean']
        true = true * train_stats['y_std'] + train_stats['y_mean']

    ade, fde = ade_fde_np(pred, true)
    print(f"Model: ADE={ade:.2f} m, FDE={fde:.2f} m")

    # Baseline on a subset for speed - use original unnormalized data
    ds_test_raw = TrajSeqDataset(args.data_npz, split="test", normalize=False)
    idx = np.random.choice(len(ds_test_raw), size=min(2000, len(ds_test_raw)), replace=False)
    Xs = ds_test_raw.X[idx]
    Ys = ds_test_raw.Y[idx]
    base_pred = np.stack([constant_velocity_baseline(Xs[i], pred_len) for i in range(len(idx))], axis=0)
    b_ade, b_fde = ade_fde_np(base_pred, Ys)
    print(f"Const-Vel Baseline: ADE={b_ade:.2f} m, FDE={b_fde:.2f} m")

    with open(os.path.join(args.output_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Model ADE: {ade:.3f}\nModel FDE: {fde:.3f}\n")
        f.write(f"Baseline ADE: {b_ade:.3f}\nBaseline FDE: {b_fde:.3f}\n")
        f.write(f"Improvement over baseline: {((b_ade - ade) / b_ade * 100):.1f}%\n")

    # Plot sample overlays in ENU
    sns.set(style="whitegrid")
    n_show = 6
    sel = np.random.choice(len(true), size=min(n_show, len(true)), replace=False)
    fig, axes = plt.subplots(2, (n_show+1)//2, figsize=(12,6))
    axes = axes.flatten()
    for j, i in enumerate(sel):
        ax = axes[j]
        # Visualize future offsets (ENU)
        ax.plot(true[i,:,0], true[i,:,1], "-o", label="True", alpha=0.8)
        ax.plot(pred[i,:,0], pred[i,:,1], "-o", label="Pred", alpha=0.8)
        ax.axhline(0, color="k", linewidth=0.5); ax.axvline(0, color="k", linewidth=0.5)
        ax.set_title(f"Sample {i}")
        ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
        if j == 0:
            ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "sample_overlays_ENU.png"), dpi=160)
    plt.close()

if __name__ == "__main__":
    main()