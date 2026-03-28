"""
Orquestrador de treino e avaliacao para todos os modelos.
Funcao principal: train_and_evaluate() — treina 1 modelo, coleta todas as metricas.
"""

import logging
import time
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import (
    TimeSeriesDataset,
    LSTMClassifier,
    TransformerClassifier,
    TCNClassifier,
    build_xgboost,
    build_logistic_regression,
    build_random_forest,
)
from .metrics import (
    classification_metrics,
    calibration_metrics,
    temporal_stability,
    prediction_distribution,
    learning_curve_data,
    permutation_importance,
)

log = logging.getLogger("Trainer")


def _split_indices(n, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return n_train, n_val, n - n_train - n_val


def _train_pytorch_model(model, X_seq, y, n_train, n_val, lr=1e-3, weight_decay=1e-4, epochs=50, batch_size=32, patience=10):
    """Treina modelo PyTorch (LSTM, Transformer, TCN) com early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = TimeSeriesDataset(X_seq[:n_train], y[:n_train])
    val_ds = TimeSeriesDataset(X_seq[n_train : n_train + n_val], y[n_train : n_train + n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            f"Epoch {epoch:>3}/{epochs} | loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.1%} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
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
    return model, history, device


def _predict_pytorch(model, X, device, batch_size=256):
    """Gera probabilidades com modelo PyTorch."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            probs.append(model(batch).cpu().numpy())
    return np.concatenate(probs)


def train_and_evaluate(
    model_name,
    data,
    model_params=None,
    train_params=None,
    compute_shap=True,
    compute_learning_curve=True,
):
    """
    Treina 1 modelo e coleta TODAS as metricas diagnosticas.

    model_name : str — "bilstm_original", "bilstm_reduced", "transformer",
                       "tcn", "xgboost", "logistic_regression", "random_forest"
    data : dict — output de load_stage*()
    model_params : dict — hiperparametros do modelo
    train_params : dict — hiperparametros de treino (lr, epochs, etc.)
    """
    if model_params is None:
        model_params = {}
    if train_params is None:
        train_params = {}

    start_time = time.time()

    sequential_models = {"bilstm_original", "bilstm_reduced", "transformer", "tcn"}
    tabular_models = {"xgboost", "logistic_regression", "random_forest"}

    is_sequential = model_name in sequential_models

    if is_sequential:
        X, y, dates = data["X_seq"], data["y_seq"], data["dates_seq"]
    else:
        X, y, dates = data["X_flat"], data["y_flat"], data["dates_flat"]

    n_train, n_val, n_test = _split_indices(len(X))
    feature_names = data["feature_names"]

    log.info(f"\n{'='*60}")
    log.info(f"Modelo: {model_name}")
    log.info(f"Split — treino: {n_train} | val: {n_val} | teste: {n_test}")
    log.info(f"{'='*60}")

    # ── Train ────────────────────────────────────────────────────────────
    history = None
    device = None

    if model_name == "bilstm_original":
        defaults = dict(hidden_size=128, n_layers=2, dropout=0.3, bidirectional=True)
        defaults.update(model_params)
        model = LSTMClassifier(input_size=X.shape[2], **defaults)
        model, history, device = _train_pytorch_model(model, X, y, n_train, n_val, **train_params)
        y_prob = _predict_pytorch(model, X[n_train + n_val :], device)

    elif model_name == "bilstm_reduced":
        defaults = dict(hidden_size=64, n_layers=1, dropout=0.2, bidirectional=True)
        defaults.update(model_params)
        model = LSTMClassifier(input_size=X.shape[2], **defaults)
        model, history, device = _train_pytorch_model(model, X, y, n_train, n_val, **train_params)
        y_prob = _predict_pytorch(model, X[n_train + n_val :], device)

    elif model_name == "transformer":
        defaults = dict(d_model=64, nhead=4, n_layers=2, dropout=0.3)
        defaults.update(model_params)
        model = TransformerClassifier(input_size=X.shape[2], **defaults)
        model, history, device = _train_pytorch_model(model, X, y, n_train, n_val, **train_params)
        y_prob = _predict_pytorch(model, X[n_train + n_val :], device)

    elif model_name == "tcn":
        defaults = dict(num_channels=[64, 64, 64], kernel_size=3, dropout=0.2, dilation_base=2)
        defaults.update(model_params)
        model = TCNClassifier(input_size=X.shape[2], **defaults)
        model, history, device = _train_pytorch_model(model, X, y, n_train, n_val, **train_params)
        y_prob = _predict_pytorch(model, X[n_train + n_val :], device)

    elif model_name == "xgboost":
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
        X_test = X[n_train + n_val :]
        model = build_xgboost(y_train, **model_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]

    elif model_name == "logistic_regression":
        X_train, y_train = X[:n_train], y[:n_train]
        X_test = X[n_train + n_val :]
        model = build_logistic_regression(y_train, **model_params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

    elif model_name == "random_forest":
        X_train, y_train = X[:n_train], y[:n_train]
        X_test = X[n_train + n_val :]
        model = build_random_forest(y_train, **model_params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    # ── Test data ────────────────────────────────────────────────────────
    y_test = y[n_train + n_val :]
    if isinstance(dates, pd.DatetimeIndex):
        dates_test = dates[n_train + n_val :]
    else:
        dates_test = dates[n_train + n_val :]

    train_time = time.time() - start_time

    # ── Metrics ──────────────────────────────────────────────────────────
    clf_metrics = classification_metrics(y_test, y_prob)
    cal_metrics = calibration_metrics(y_test, y_prob)
    temp_metrics = temporal_stability(y_test, y_prob, dates_test)
    pred_dist = prediction_distribution(y_test, y_prob)

    log.info(f"AUC: {clf_metrics['roc_auc']:.4f} | Acc: {clf_metrics['accuracy']:.1%} | "
             f"F1: {clf_metrics['f1']:.4f} | ECE: {cal_metrics['ece']:.4f} | "
             f"Brier: {cal_metrics['brier_score']:.4f}")
    log.info(clf_metrics["report"])

    # ── SHAP (tabular models only) ───────────────────────────────────────
    shap_values = None
    if compute_shap and model_name in tabular_models:
        try:
            import shap
            X_test_tab = X[n_train + n_val :]
            if model_name == "xgboost":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_tab)
            elif model_name in ("logistic_regression", "random_forest"):
                explainer = shap.Explainer(model, X[:n_train])
                shap_values = explainer(X_test_tab).values
        except Exception as e:
            log.warning(f"SHAP falhou para {model_name}: {e}")

    # ── Permutation importance (tabular) ─────────────────────────────────
    perm_imp = None
    if model_name in tabular_models:
        X_test_tab = X[n_train + n_val :]
        def predict_fn(x):
            return model.predict_proba(x)[:, 1]
        perm_imp, baseline_auc = permutation_importance(
            predict_fn, X_test_tab, y_test, feature_names, n_repeats=5
        )

    # ── Learning curve (tabular) ─────────────────────────────────────────
    lc_data = None
    if compute_learning_curve and model_name in tabular_models:
        X_train_tab = X[:n_train]
        y_train_tab = y[:n_train]
        X_test_tab = X[n_train + n_val :]

        def lc_train_fn(X_tr, y_tr, X_te):
            if model_name == "xgboost":
                m = build_xgboost(y_tr, **model_params)
                m.set_params(early_stopping_rounds=None)
                m.fit(X_tr, y_tr)
                return m.predict_proba(X_te)[:, 1]
            elif model_name == "logistic_regression":
                m = build_logistic_regression(y_tr, **model_params)
                m.fit(X_tr, y_tr)
                return m.predict_proba(X_te)[:, 1]
            elif model_name == "random_forest":
                m = build_random_forest(y_tr, **model_params)
                m.fit(X_tr, y_tr)
                return m.predict_proba(X_te)[:, 1]

        lc_data = learning_curve_data(lc_train_fn, X_train_tab, y_train_tab, X_test_tab, y_test)

    # ── Result ───────────────────────────────────────────────────────────
    result = {
        "model_name": model_name,
        "model_params": model_params,
        "train_params": train_params,
        "train_time_seconds": train_time,
        "history": history,
        "y_true": y_test,
        "y_prob": y_prob,
        "dates_test": dates_test,
        "classification": clf_metrics,
        "calibration": cal_metrics,
        "temporal_stability": temp_metrics,
        "prediction_distribution": pred_dist,
        "shap_values": shap_values,
        "permutation_importance": perm_imp,
        "learning_curve": lc_data,
        "model": model,
        "device": device,
    }

    return result


def save_results_json(results, path):
    """Salva metricas numericas em JSON (sem arrays numpy)."""
    serializable = {}
    for name, r in results.items():
        serializable[name] = {
            "model_name": r["model_name"],
            "model_params": r["model_params"],
            "train_params": r["train_params"],
            "train_time_seconds": r["train_time_seconds"],
            "roc_auc": r["classification"]["roc_auc"],
            "accuracy": r["classification"]["accuracy"],
            "f1": r["classification"]["f1"],
            "precision": r["classification"]["precision"],
            "recall": r["classification"]["recall"],
            "f1_desce": r["classification"]["f1_desce"],
            "precision_desce": r["classification"]["precision_desce"],
            "recall_desce": r["classification"]["recall_desce"],
            "ece": r["calibration"]["ece"],
            "brier_score": r["calibration"]["brier_score"],
        }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    log.info(f"Resultados salvos em {path}")
