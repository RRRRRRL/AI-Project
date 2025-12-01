import torch
import torch.nn as nn

class LSTMMultiHorizon(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.2, pred_len=20):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len * 3),
        )
        self.pred_len = pred_len

    def forward(self, x):
        # x: [B, H, F]
        out, (h_n, c_n) = self.lstm(x)  # out: [B, H, hidden]; take last step
        last = out[:, -1, :]  # [B, hidden]
        y = self.head(last)  # [B, K*3]
        y = y.view(-1, self.pred_len, 3)  # [B, K, 3] offsets in ENU
        return y