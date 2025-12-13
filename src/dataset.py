import numpy as np
import torch
from torch.utils.data import Dataset

class TrajSeqDataset(Dataset):
    def __init__(self, npz_path, split="train", train_ratio=0.7, val_ratio=0.15, seed=42, 
                 normalize=True, stats=None):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]  # [N, H, F]
        self.Y = data["Y"]  # [N, K, 3]
        self.flights = data["flights"]
        self.normalize = normalize
        
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
        
        # Compute or use normalization statistics
        if self.normalize:
            if stats is None and split == "train":
                # Compute stats from training data
                self.x_mean = np.mean(self.X, axis=(0, 1), keepdims=True)
                self.x_std = np.std(self.X, axis=(0, 1), keepdims=True) + 1e-8
                self.y_mean = np.mean(self.Y, axis=(0, 1), keepdims=True)
                self.y_std = np.std(self.Y, axis=(0, 1), keepdims=True) + 1e-8
            elif stats is not None:
                # Use provided stats (for val/test)
                self.x_mean = stats['x_mean']
                self.x_std = stats['x_std']
                self.y_mean = stats['y_mean']
                self.y_std = stats['y_std']
            else:
                # Val/test without stats - compute from own data (less ideal)
                self.x_mean = np.mean(self.X, axis=(0, 1), keepdims=True)
                self.x_std = np.std(self.X, axis=(0, 1), keepdims=True) + 1e-8
                self.y_mean = np.mean(self.Y, axis=(0, 1), keepdims=True)
                self.y_std = np.std(self.Y, axis=(0, 1), keepdims=True) + 1e-8
            
            # Apply normalization
            self.X = (self.X - self.x_mean) / self.x_std
            self.Y = (self.Y - self.y_mean) / self.y_std
    
    def get_stats(self):
        """Return normalization statistics for use in val/test sets"""
        if self.normalize:
            return {
                'x_mean': self.x_mean,
                'x_std': self.x_std,
                'y_mean': self.y_mean,
                'y_std': self.y_std
            }
        return None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # [H, F]
        y = torch.from_numpy(self.Y[idx])  # [K, 3]
        return x, y