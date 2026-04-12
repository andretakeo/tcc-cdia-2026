"""
Funcoes para calcular todas as metricas diagnosticas.
Cada funcao retorna dados estruturados (dict ou arrays) para plotagem posterior.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
)


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_desce": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_desce": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_desce": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=["Desce", "Sobe"]),
        "y_pred": y_pred,
    }


def calibration_metrics(y_true, y_prob, n_bins=10):
    brier = brier_score_loss(y_true, y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
            continue
        bin_accs.append(y_true[mask].mean())
        bin_confs.append(y_prob[mask].mean())
        bin_counts.append(mask.sum())

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    total = bin_counts.sum()
    ece = np.sum(bin_counts / total * np.abs(bin_accs - bin_confs)) if total > 0 else 0

    return {
        "brier_score": brier,
        "ece": ece,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges,
    }


def temporal_stability(y_true, y_prob, dates, window_months=3):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}, index=dates)
    results = []
    window_start = df.index[0]
    window_delta = pd.DateOffset(months=window_months)

    while window_start + window_delta <= df.index[-1]:
        window_end = window_start + window_delta
        mask = (df.index >= window_start) & (df.index < window_end)
        subset = df[mask]

        if len(subset) >= 10 and subset["y_true"].nunique() == 2:
            auc = roc_auc_score(subset["y_true"], subset["y_prob"])
        else:
            auc = np.nan

        results.append({"start": window_start, "end": window_end, "auc": auc, "n_samples": len(subset)})
        window_start += pd.DateOffset(months=1)

    return pd.DataFrame(results)


def prediction_distribution(y_true, y_prob, n_bins=30):
    return {
        "probs_sobe": y_prob[y_true == 1],
        "probs_desce": y_prob[y_true == 0],
        "n_bins": n_bins,
    }


def learning_curve_data(train_fn, X_train, y_train, X_test, y_test, fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    results = []
    n = len(X_train)
    for frac in fractions:
        n_subset = int(n * frac)
        if n_subset < 10:
            continue
        X_sub = X_train[:n_subset]
        y_sub = y_train[:n_subset]
        y_prob = train_fn(X_sub, y_sub, X_test)
        auc = roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else np.nan
        results.append({"fraction": frac, "n_samples": n_subset, "auc": auc})

    return pd.DataFrame(results)


def permutation_importance(model_predict_fn, X_test, y_test, feature_names, n_repeats=5):
    baseline_auc = roc_auc_score(y_test, model_predict_fn(X_test))
    importances = {}

    for i, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, i])
            perm_auc = roc_auc_score(y_test, model_predict_fn(X_perm))
            drops.append(baseline_auc - perm_auc)
        importances[name] = {"mean_drop": np.mean(drops), "std_drop": np.std(drops)}

    return importances, baseline_auc
