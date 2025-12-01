import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_curves(history, outdir):
    # history: dict with lists: train_loss, val_loss, train_f1, val_f1, steps or epochs
    sns.set(style="whitegrid")
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_f1"], label="train_f1")
    plt.plot(epochs, history["val_f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "f1_curve.png"), dpi=160)
    plt.close()