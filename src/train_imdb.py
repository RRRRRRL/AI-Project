import os
import argparse
import math
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AdamW,
                          get_linear_schedule_with_warmup, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm
from utils import set_seed, ensure_dir, save_dict_as_csv
from plotting import plot_curves

@dataclass
class Args:
    seed: int = 42
    model_name: str = "distilbert-base-uncased"
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_length: int = 256
    grad_clip_norm: float = 1.0
    output_dir: str = "outputs/run1"
    patience: int = 3

def compute_metrics(preds, labels):
    # preds: [N, 2] logits
    probs = torch.softmax(torch.tensor(preds), dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    y_true = labels
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs[:,1])
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="outputs/run1")
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "plots"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Load dataset
    dataset = load_dataset("imdb")  # splits: train (25k), test (25k); create validation from train
    split = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = dataset["test"]

    # Tokenizer and encoding
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(preprocess, batched=True, remove_columns=["text"])

    # DataLoaders
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    # Optimizer/Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    num_training_steps = args.epochs * math.ceil(len(train_loader))
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_val_f1 = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits.detach().cpu().numpy()
            train_preds.append(logits)
            train_labels.append(labels.detach().cpu().numpy())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader)
        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_metrics = compute_metrics(train_preds, train_labels)

        # Validate
        model.eval()
        val_running_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits.detach().cpu().numpy()

                val_running_loss += loss.item()
                val_preds.append(logits)
                val_labels.append(labels.detach().cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_metrics = compute_metrics(val_preds, val_labels)

        # Logging
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['roc_auc']:.4f}")

        # Early stopping & save best
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "checkpoint-best")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

        # Save per-epoch metrics
        save_dict_as_csv(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_roc_auc": val_metrics["roc_auc"],
            },
            os.path.join(args.output_dir, f"metrics_epoch_{epoch}.csv"),
        )

    # Final plots
    plot_curves(history, os.path.join(args.output_dir, "plots"))

    # Test set evaluation with best checkpoint
    print("Loading best checkpoint for test evaluation...")
    best_model_dir = os.path.join(args.output_dir, "checkpoint-best")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
    best_model.eval()
    test_preds, test_labels = [], []
    test_running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = best_model(**batch)
            loss = outputs.loss
            logits = outputs.logits.detach().cpu().numpy()
            test_running_loss += loss.item()
            test_preds.append(logits)
            test_labels.append(labels.detach().cpu().numpy())
    test_loss = test_running_loss / len(test_loader)
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_metrics = compute_metrics(test_preds, test_labels)

    # Confusion Matrix
    y_pred = test_preds.argmax(axis=1)
    cm = confusion_matrix(test_labels, y_pred)
    print("Test metrics:", test_metrics)
    print("Confusion matrix:\n", cm)

    # Save summary
    save_dict_as_csv(
        {
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
        },
        os.path.join(args.output_dir, "test_metrics.csv"),
    )

if __name__ == "__main__":
    main()