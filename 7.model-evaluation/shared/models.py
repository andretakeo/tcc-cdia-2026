"""
7 modelos para classificacao binaria sobe/desce.
Todos seguem a mesma interface:
  - Modelos PyTorch: __init__(input_size, **kwargs), forward(x) -> probabilidades
  - Modelos sklearn: instanciados com parametros, .fit(), .predict_proba()
"""

import math
import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, n_layers: int = 2, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0.0, bidirectional=bidirectional, batch_first=True)
        direction_factor = 2 if bidirectional else 1
        lstm_out = hidden_size * direction_factor
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_out, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(1)


class CausalConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = padding
        self.chomp2 = padding
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        if self.chomp1 > 0:
            out = out[:, :, :-self.chomp1]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.chomp2 > 0:
            out = out[:, :, :-self.chomp2]
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, input_size: int, num_channels: list = None, kernel_size: int = 3, dropout: float = 0.2, dilation_base: int = 2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            dilation = dilation_base ** i
            layers.append(CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(num_channels[-1], 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.mean(dim=2)
        return self.head(x).squeeze(1)


def build_xgboost(y_train, **kwargs):
    defaults = dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", early_stopping_rounds=20, random_state=42, use_label_encoder=False)
    defaults["scale_pos_weight"] = (1 - y_train.mean()) / y_train.mean()
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def build_logistic_regression(y_train, **kwargs):
    defaults = dict(C=1.0, penalty="l2", solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=42)
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def build_random_forest(y_train, **kwargs):
    defaults = dict(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", bootstrap=True, random_state=42, n_jobs=-1)
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)
