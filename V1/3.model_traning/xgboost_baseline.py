"""
Baseline XGBoost — Classificação binária sobe/desce em 21 dias
─────────────────────────────────────────────────────────────────────────────
Modelo clássico (sem dependência temporal) para comparação com o BiLSTM.
Usa as mesmas features e o mesmo split temporal walk-forward para garantir
comparação justa.

Dependências:
    pip install xgboost scikit-learn numpy pandas matplotlib
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
log = logging.getLogger("XGBoostBaseline")


def build_flat_dataset(
    X_full: pd.DataFrame,
    horizon: int = 21,
    pca_components: int = 32,
) -> tuple:
    """
    Prepara dataset tabular (sem janelas temporais) para XGBoost.
    Usa o mesmo PCA e normalização do pipeline LSTM para comparação justa.
    """
    log.info(f"Shape de entrada: {X_full.shape}")

    close = X_full["Close"]
    target = (close.shift(-horizon) > close).astype(int)
    target = target.iloc[:-horizon]
    X = X_full.iloc[:-horizon].copy()

    log.info(f"Distribuição do target: {target.value_counts().to_dict()}  "
             f"(balance: {target.mean():.1%} sobe)")

    emb_cols = [c for c in X.columns if c.startswith("emb_")]
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

    n_train = int(len(X) * 0.7)
    scaler = StandardScaler()
    scaler.fit(X.iloc[:n_train])
    X_scaled = scaler.transform(X)

    return X_scaled, target.values, X.index, feature_names


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple:
    """Treina XGBoost com split cronológico walk-forward."""
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    log.info(f"Split — treino: {n_train} | val: {n_val} | teste: {n - n_train - n_val}")
    log.info(f"Treino: {dates[0].date()} → {dates[n_train-1].date()}")
    log.info(f"Val:    {dates[n_train].date()} → {dates[n_train+n_val-1].date()}")
    log.info(f"Teste:  {dates[n_train+n_val].date()} → {dates[-1].date()}")

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    scale_pos_weight = (1 - y_train.mean()) / y_train.mean()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    log.info(f"Melhor iteração: {model.best_iteration}")
    return model, X_test, y_test, dates[n_train+n_val:]


def evaluate_xgboost(model, X_test, y_test):
    """Métricas completas no conjunto de teste."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    log.info("\n" + classification_report(y_test, preds, target_names=["Desce", "Sobe"]))
    auc = roc_auc_score(y_test, probs)
    log.info(f"ROC-AUC: {auc:.4f}")

    return probs, preds, auc


def plot_roc(probs, y_test, save_path="xgboost_roc.png"):
    """Plota e salva curva ROC."""
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"XGBoost AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("Curva ROC — XGBoost Baseline")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info(f"Curva ROC salva em {save_path}")
    plt.show()
