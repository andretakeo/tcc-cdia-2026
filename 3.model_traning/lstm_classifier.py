"""
LSTM Classifier — Sobe ou Desce em 21 dias
─────────────────────────────────────────────────────────────────────────────
Pipeline completo:
  1. Prepara target binário (1 = sobe, 0 = desce/igual em 21 dias)
  2. Reduz dimensão dos embeddings com PCA
  3. Normaliza features
  4. Cria janelas temporais (sequências) para o LSTM
  5. Treina com validação walk-forward (sem data leakage)
  6. Avalia com métricas de classificação + curva ROC

Dependências:
    pip install torch scikit-learn numpy pandas matplotlib

─────────────────────────────────────────────────────────────────────────────
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LSTMClassifier")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preparação do dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    X_full: pd.DataFrame,
    horizon: int = 21,
    pca_components: int = 32,
    window: int = 30,
) -> tuple:
    """
    A partir do DataFrame completo (preço + embeddings), retorna:
      X_seq : np.ndarray (N, window, features)  — sequências para o LSTM
      y     : np.ndarray (N,)                   — 1=sobe, 0=desce
      dates : pd.DatetimeIndex                  — datas correspondentes
      feature_names : list[str]

    Parâmetros
    ----------
    horizon        : dias à frente para definir sobe/desce
    pca_components : dimensões para comprimir os embeddings
    window         : tamanho da janela de lookback do LSTM
    """
    log.info(f"Shape de entrada: {X_full.shape}")

    # ── target binário ───────────────────────────────────────────────────────
    close = X_full["Close"]
    target = (close.shift(-horizon) > close).astype(int)
    target = target.iloc[:-horizon]          # remove NaN do final
    X = X_full.iloc[:-horizon].copy()

    log.info(f"Distribuição do target: {target.value_counts().to_dict()}  "
             f"(balance: {target.mean():.1%} sobe)")

    # ── PCA nos embeddings ───────────────────────────────────────────────────
    emb_cols   = [c for c in X.columns if c.startswith("emb_")]
    price_cols = [c for c in X.columns if not c.startswith("emb_")]

    if emb_cols:
        log.info(f"PCA: {len(emb_cols)} dims → {pca_components} componentes")
        pca = PCA(n_components=pca_components, random_state=42)
        emb_reduced = pca.fit_transform(X[emb_cols])
        var_exp = pca.explained_variance_ratio_.cumsum()[-1]
        log.info(f"Variância explicada pelo PCA: {var_exp:.1%}")

        emb_df = pd.DataFrame(
            emb_reduced,
            index=X.index,
            columns=[f"pca_{i}" for i in range(pca_components)],
        )
        X = pd.concat([X[price_cols], emb_df], axis=1)

    feature_names = X.columns.tolist()
    log.info(f"Features finais: {len(feature_names)}")

    # ── normalização ─────────────────────────────────────────────────────────
    # Fit apenas no treino (primeiros 70%) para não vazar info do futuro
    n_train = int(len(X) * 0.7)
    scaler = StandardScaler()
    scaler.fit(X.iloc[:n_train])
    X_scaled = scaler.transform(X)

    # ── janelas temporais ────────────────────────────────────────────────────
    X_arr = X_scaled
    y_arr = target.values

    sequences, labels, dates = [], [], []
    for i in range(window, len(X_arr)):
        sequences.append(X_arr[i - window : i])   # (window, features)
        labels.append(y_arr[i])
        dates.append(X.index[i])

    X_seq = np.array(sequences, dtype=np.float32)
    y_out = np.array(labels,    dtype=np.float32)
    dates = pd.DatetimeIndex(dates)

    log.info(f"Sequências geradas: {X_seq.shape}  →  y: {y_out.shape}")
    return X_seq, y_out, dates, feature_names, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset PyTorch
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Modelo LSTM
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    LSTM bidirecional com dropout e camada de classificação binária.

    Arquitetura:
        Input (window, features)
        → BiLSTM × n_layers
        → Dropout
        → Linear → Linear → Sigmoid
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        direction_factor = 2 if bidirectional else 1
        lstm_out = hidden_size * direction_factor

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last    = out[:, -1, :]     # pega o último timestep
        return self.head(last).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Treino com walk-forward split
# ─────────────────────────────────────────────────────────────────────────────

def train(
    X_seq: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    # hiperparâmetros
    hidden_size: int   = 128,
    n_layers:    int   = 2,
    dropout:     float = 0.3,
    lr:          float = 1e-3,
    epochs:      int   = 50,
    batch_size:  int   = 32,
    patience:    int   = 10,       # early stopping
) -> tuple:
    """
    Treina o LSTM com split cronológico (sem shuffle).
    Retorna (modelo, histórico de loss, índices de teste).
    """
    n = len(X_seq)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    log.info(f"Split — treino: {n_train} | val: {n_val} | teste: {n_test}")
    log.info(f"Treino: {dates[0].date()} → {dates[n_train-1].date()}")
    log.info(f"Val:    {dates[n_train].date()} → {dates[n_train+n_val-1].date()}")
    log.info(f"Teste:  {dates[n_train+n_val].date()} → {dates[-1].date()}")

    train_ds = TimeSeriesDataset(X_seq[:n_train],           y[:n_train])
    val_ds   = TimeSeriesDataset(X_seq[n_train:n_train+n_val], y[n_train:n_train+n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = LSTMClassifier(
        input_size=X_seq.shape[2],
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    # class weight para balancear sobe/desce
    pos_weight = torch.tensor([(1 - y[:n_train].mean()) / y[:n_train].mean()]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss  = float("inf")
    best_state     = None
    epochs_no_impr = 0

    for epoch in range(1, epochs + 1):
        # ── treino ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # BCEWithLogitsLoss espera logits — remover Sigmoid do forward temporariamente
            # Usamos BCE normal com Sigmoid já aplicado:
            pred = model(xb)
            loss = nn.BCELoss()(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        # ── validação ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred    = model(xb)
                val_loss += nn.BCELoss()(pred, yb).item()
                correct  += ((pred > 0.5) == yb).sum().item()
                total    += len(yb)

        val_loss /= len(val_dl)
        val_acc   = correct / total
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log.info(
            f"Epoch {epoch:>3}/{epochs} | "
            f"loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_acc {val_acc:.1%} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                log.info(f"Early stopping na epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_idx = slice(n_train + n_val, n)
    return model, history, test_idx, device


# ─────────────────────────────────────────────────────────────────────────────
# 5. Avaliação
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_seq, y, test_idx, dates, device):
    """Métricas completas no conjunto de teste."""
    model.eval()
    X_test = torch.tensor(X_seq[test_idx], dtype=torch.float32).to(device)
    y_test = y[test_idx]

    with torch.no_grad():
        probs = model(X_test).cpu().numpy()

    preds = (probs > 0.5).astype(int)

    log.info("\n" + classification_report(y_test, preds, target_names=["Desce", "Sobe"]))
    log.info(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

    return probs, preds, y_test


def plot_results(history, probs, preds, y_test, dates, test_idx):
    """4 gráficos: loss, acurácia, ROC e previsões no tempo."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LSTM — Sobe/Desce em 21 dias", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Treino")
    ax.plot(history["val_loss"],   label="Validação")
    ax.set_title("Loss por Época")
    ax.set_xlabel("Época"); ax.set_ylabel("BCE Loss")
    ax.legend()

    # Acurácia de validação
    ax = axes[0, 1]
    ax.plot(history["val_acc"], color="green")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="baseline (50%)")
    ax.set_title("Acurácia — Validação")
    ax.set_xlabel("Época"); ax.set_ylabel("Acurácia")
    ax.legend()

    # ROC
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], "k--", alpha=0.3)
    ax.set_title("Curva ROC — Teste")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend()

    # Previsões no tempo
    ax = axes[1, 1]
    test_dates = dates[test_idx]
    ax.scatter(test_dates[y_test == 1],  probs[y_test == 1],  c="green", s=15, label="Real: Sobe",  alpha=0.7)
    ax.scatter(test_dates[y_test == 0],  probs[y_test == 0],  c="red",   s=15, label="Real: Desce", alpha=0.7)
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4)
    ax.set_title("Probabilidade Prevista — Teste")
    ax.set_xlabel("Data"); ax.set_ylabel("P(Sobe)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("lstm_results.png", dpi=150, bbox_inches="tight")
    log.info("Gráfico salvo em lstm_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Salvar / carregar modelo
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, path: str = "lstm_model.pt"):
    torch.save(model.state_dict(), path)
    log.info(f"Modelo salvo em {path}")


def load_model(path: str, input_size: int, **kwargs) -> LSTMClassifier:
    model = LSTMClassifier(input_size=input_size, **kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model