import numpy as np
import torch
from torch.utils.data import Dataset

class TrajSeqDataset(Dataset):
    def __init__(self, npz_path, split="train", train_ratio=0.7, val_ratio=0.15, seed=42):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]  # [N, H, F]
        self.Y = data["Y"]  # [N, K, 3]
        self.flights = data["flights"]
        # Split by flights to avoid leakage
        uniq = np.unique(self.flights)
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n_train = int(len(uniq) * train_ratio)
        n_val = int(len(uniq) * val_ratio)
        train_f = set(uniq[:n_train])
        val_f = set(uniq[n_train:n_train+n_val])
        test_f = set(uniq[n_train+n_val:])

        if split == "train":
            mask = np.isin(self.flights, list(train_f))
        elif split == "val":
            mask = np.isin(self.flights, list(val_f))
        else:
            mask = np.isin(self.flights, list(test_f))
        self.X = self.X[mask]
        self.Y = self.Y[mask]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # [H, F]
        y = torch.from_numpy(self.Y[idx])  # [K, 3]
        return x, y