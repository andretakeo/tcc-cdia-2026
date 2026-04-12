"""
Funcoes de visualizacao para todas as metricas diagnosticas.
Cada funcao recebe dados estruturados de metrics.py e gera graficos.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay


def plot_roc_curves(results, title="Curvas ROC", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        auc = roc_auc_score(r["y_true"], r["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Aleatorio (AUC=0.500)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrices(results, save_path=None):
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (name, r) in zip(axes, results.items()):
        ConfusionMatrixDisplay(confusion_matrix=r["confusion_matrix"], display_labels=["Desce", "Sobe"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontsize=11)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Matrizes de Confusao", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_calibration_diagrams(results, save_path=None):
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (name, cal) in zip(axes, results.items()):
        mask = cal["bin_counts"] > 0
        ax.plot(cal["bin_confs"][mask], cal["bin_accs"][mask], "s-", label=f"ECE={cal['ece']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(f"{name}\nBrier={cal['brier_score']:.3f}", fontsize=10)
        ax.set_xlabel("Confianca prevista")
        ax.set_ylabel("Fracao de positivos")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Diagramas de Confiabilidade", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_temporal_stability(results, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, df in results.items():
        ax.plot(df["start"], df["auc"], "o-", label=name, markersize=4)

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Aleatorio (0.5)")
    ax.set_title("Estabilidade Temporal — AUC por janela de 3 meses", fontsize=14, fontweight="bold")
    ax.set_xlabel("Inicio da janela")
    ax.set_ylabel("ROC-AUC")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_prediction_distributions(results, save_path=None):
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (name, dist) in zip(axes, results.items()):
        ax.hist(dist["probs_desce"], bins=dist["n_bins"], alpha=0.5, color="red", label="Desce", density=True)
        ax.hist(dist["probs_sobe"], bins=dist["n_bins"], alpha=0.5, color="green", label="Sobe", density=True)
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.4)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("P(Sobe)")
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Distribuicao de Previsoes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_learning_curves(results, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in results.items():
        ax.plot(df["fraction"] * 100, df["auc"], "o-", label=name)

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Aleatorio (0.5)")
    ax.set_title("Curvas de Aprendizado — AUC vs % dos dados de treino", fontsize=14, fontweight="bold")
    ax.set_xlabel("% dos dados de treino")
    ax.set_ylabel("ROC-AUC no teste")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(importances, baseline_auc, title="Importancia por Permutacao", save_path=None):
    names = list(importances.keys())
    means = [importances[n]["mean_drop"] for n in names]
    stds = [importances[n]["std_drop"] for n in names]

    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    ax.barh(names, means, xerr=stds, color="steelblue", alpha=0.8)
    ax.set_xlabel(f"Queda no AUC (baseline={baseline_auc:.3f})")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_summary(shap_values, feature_names, title="SHAP Summary", save_path=None):
    import shap
    fig = plt.figure(figsize=(10, max(4, len(feature_names) * 0.3)))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attention_heatmap(attention_weights, title="Transformer Attention Weights", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dia (chave)")
    ax.set_ylabel("Dia (query)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
