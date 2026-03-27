"""
Transformer Classifier — Sobe ou Desce em 21 dias
─────────────────────────────────────────────────────────────────────────────
Modelo alternativo ao BiLSTM usando atenção multi-cabeça para comparação.
Usa o mesmo dataset, split walk-forward e métricas do pipeline LSTM.

Arquitetura:
    Input (window, features) → Positional Encoding
    → TransformerEncoder(2 camadas, d_model=64, nhead=4, dropout=0.3)
    → Mean pooling temporal → Dense(32) → ReLU → Dense(1) → Sigmoid

Dependências:
    pip install torch scikit-learn numpy pandas matplotlib
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TransformerClassifier")


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding padrão (sinusoidal) para sequências temporais."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer simples com atenção multi-cabeça para classificação binária.

    Arquitetura:
        Linear(input_size → d_model) → PositionalEncoding
        → TransformerEncoder(n_layers, nhead, d_ff=d_model*4)
        → Mean pooling → Dense(32) → ReLU → Dropout → Dense(1) → Sigmoid
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # mean pooling temporal
        return self.head(x).squeeze(1)


def train_transformer(
    X_seq: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    d_model: int = 64,
    nhead: int = 4,
    n_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
) -> tuple:
    """Treina o Transformer com split cronológico walk-forward."""
    n = len(X_seq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    log.info(f"Split — treino: {n_train} | val: {n_val} | teste: {n_test}")

    train_ds = TimeSeriesDataset(X_seq[:n_train], y[:n_train])
    val_ds = TimeSeriesDataset(X_seq[n_train:n_train+n_val], y[n_train:n_train+n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = TransformerClassifier(
        input_size=X_seq.shape[2],
        d_model=d_model,
        nhead=nhead,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_impr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.BCELoss()(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += nn.BCELoss()(pred, yb).item()
                correct += ((pred > 0.5) == yb).sum().item()
                total += len(yb)

        val_loss /= len(val_dl)
        val_acc = correct / total
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log.info(
            f"Epoch {epoch:>3}/{epochs} | "
            f"loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_acc {val_acc:.1%} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                log.info(f"Early stopping na epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_idx = slice(n_train + n_val, n)
    return model, history, test_idx, device


def evaluate_transformer(model, X_seq, y, test_idx, dates, device):
    """Métricas completas no conjunto de teste."""
    model.eval()
    X_test = torch.tensor(X_seq[test_idx], dtype=torch.float32).to(device)
    y_test = y[test_idx]

    with torch.no_grad():
        probs = model(X_test).cpu().numpy()

    preds = (probs > 0.5).astype(int)

    log.info("\n" + classification_report(y_test, preds, target_names=["Desce", "Sobe"]))
    auc = roc_auc_score(y_test, probs)
    log.info(f"ROC-AUC: {auc:.4f}")

    return probs, preds, y_test, auc


def plot_transformer_results(history, probs, y_test, save_path="transformer_results.png"):
    """Plota loss e curva ROC do Transformer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Transformer — Sobe/Desce em 21 dias", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(history["train_loss"], label="Treino")
    ax.plot(history["val_loss"], label="Validação")
    ax.set_title("Loss por Época")
    ax.set_xlabel("Época")
    ax.set_ylabel("BCE Loss")
    ax.legend()

    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("Curva ROC — Teste")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info(f"Gráficos salvos em {save_path}")
    plt.show()
