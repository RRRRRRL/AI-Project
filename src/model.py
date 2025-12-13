import torch
import torch.nn as nn

class LSTMMultiHorizon(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.2, pred_len=20, 
                 use_layer_norm=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Improved head with layer normalization and better architecture
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_len * 3),
        )
        self.pred_len = pred_len

    def forward(self, x):
        # x: [B, H, F]
        out, (h_n, c_n) = self.lstm(x)  # out: [B, H, hidden]; take last step
        last = out[:, -1, :]  # [B, hidden]
        if self.use_layer_norm:
            last = self.layer_norm(last)
        y = self.head(last)  # [B, K*3]
        y = y.view(-1, self.pred_len, 3)  # [B, K, 3] offsets in ENU
        return y