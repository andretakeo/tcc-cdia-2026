"""
Utilitários compartilhados para avaliação rigorosa de modelos.

Inclui:
- bootstrap_auc_ci: intervalo de confiança bootstrap para ROC-AUC
- walk_forward_split: split temporal 70/15/15 sem leakage
- evaluate_model: métricas padronizadas (AUC, accuracy, F1, confusion)
- format_metric_with_ci: formatação "0.71 [0.64, 0.78]"

Estes utilitários são reusados por:
- 9.baselines/dumb_baseline.ipynb (baseline autoregressivo sem sentimento)
- experimentos multi-ticker (PETR4, VALE3) na Etapa 10
- regime split (Etapa 11)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def walk_forward_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal cronológico 70/15/15. Sem shuffle.

    Pressupõe que `df` já está ordenado por data crescente.
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Intervalo de confiança bootstrap para ROC-AUC.

    Reamostra com reposição (n_boot vezes) os pares (y_true, y_score)
    e calcula o AUC em cada reamostragem. Retorna (auc_pontual, lower, upper)
    com intervalo de confiança alpha.

    Use isto SEMPRE que reportar AUC. Um AUC sem CI é uma alegação sem
    margem de erro — e em datasets pequenos (~180 amostras de teste) a
    variância é grande o suficiente para que diferenças aparentes sejam
    estatisticamente nulas.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Garante que há ao menos uma amostra de cada classe
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], y_score[idx])
    aucs = aucs[~np.isnan(aucs)]
    auc_point = roc_auc_score(y_true, y_score)
    lower = float(np.quantile(aucs, alpha / 2))
    upper = float(np.quantile(aucs, 1 - alpha / 2))
    return float(auc_point), lower, upper


def format_metric_with_ci(point: float, lower: float, upper: float) -> str:
    """Formata 'AUC = 0.71 [0.64, 0.78]'."""
    return f"{point:.3f} [{lower:.3f}, {upper:.3f}]"


def evaluate_model(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 1000,
) -> dict:
    """
    Avaliação padronizada de um classificador binário probabilístico.

    Retorna dict com: auc, auc_ci, accuracy, f1_pos, f1_neg, confusion,
    classification_report, n_test, class_balance.
    """
    y_pred = (y_score >= threshold).astype(int)
    auc, lo, hi = bootstrap_auc_ci(y_true, y_score, n_boot=n_boot)
    return {
        "auc": auc,
        "auc_ci": (lo, hi),
        "auc_str": format_metric_with_ci(auc, lo, hi),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_neg": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
        "n_test": int(len(y_true)),
        "class_balance": float(np.mean(y_true)),
    }


def make_binary_target(close: pd.Series, horizon: int = 21) -> pd.Series:
    """
    Target binário: 1 se Close[t+horizon] > Close[t], senão 0.
    Os últimos `horizon` valores ficam NaN e devem ser dropados.
    """
    future = close.shift(-horizon)
    target = (future > close).astype(float)
    target[future.isna()] = np.nan
    return target
