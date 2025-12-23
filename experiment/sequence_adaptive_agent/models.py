import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, x):
        # logits for giving reward to TARGET (vs anti)
        return self.net(x).squeeze(-1)


class LSTMPolicy(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx=None):
        # x: (batch, seq_len, input_dim)
        out, h = self.lstm(x, hx)
        # take last time step
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1), h


class GRUPolicy(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim,
                          hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx=None):
        out, h = self.gru(x, hx)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1), h


class BiLSTMPolicy(nn.Module):
    """Dual-head BiLSTM policy.

    Given an input sequence of shape (batch, seq_len, input_dim), outputs two
    independent logits (batch, 2):
    - logit[0]: probability of allocating reward to target side
    - logit[1]: probability of allocating reward to anti-target side

    The calling code may enforce exact per-side budgets (25/100) externally.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if
            (num_layers > 1 and dropout and dropout > 0.0) else 0.0,
        )
        self.dropout = nn.Dropout(
            dropout) if dropout and dropout > 0.0 else None
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x, hx=None):
        out, h = self.lstm(x, hx)
        last = out[:, -1, :]
        if self.dropout is not None:
            last = self.dropout(last)
        return self.fc(last), h


class ConvPolicy(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, kernel_size=3):
        super().__init__()
        # input: (batch, seq_len, input_dim) -> conv1d requires (batch, channels, seq_len)
        self.conv = nn.Conv1d(in_channels=input_dim,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx=None):
        # x: (batch, seq_len, input_dim)
        x_t = x.permute(0, 2, 1)
        c = F.relu(self.conv(x_t))
        p = self.pool(c).squeeze(-1)
        return self.fc(p).squeeze(-1), None


class TransformerPolicy(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, nhead=4, nlayers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx=None):
        # x: (batch, seq_len, input_dim) -> Transformer expects (seq_len, batch, d_model)
        z = self.input_proj(x)
        z = z.permute(1, 0, 2)
        out = self.encoder(z)
        last = out[-1]
        return self.fc(last).squeeze(-1), None
