# 01 — Project Overview

## Research Question

**Do Brazilian financial news improve stock price direction prediction (up/down) for B3 equities?**

This decomposes into two sub-questions:

1. **Textual representation**: Do domain-specific sentiment features (FinBERT-PT-BR, 5 dimensions) outperform generic high-dimensional embeddings (Ollama, 1,024 dimensions)?
2. **Methodological robustness**: Do results obtained under standard single-window evaluation survive expanding-window cross-validation, multi-seed analysis, and formal statistical tests?

## Stocks Studied

| Ticker | Company | Sector | Articles Collected |
|---|---|---|---:|
| ITUB4 | Itaú Unibanco | Banking | 2,572 |
| PETR4 | Petrobras | Oil & Gas | 1,775 |
| VALE3 | Vale | Mining | 1,525 |

All are large-cap B3 equities with high liquidity. Period: 2009–2026 (news), 2021–2026 (features/models).

## The Narrative Arc

### Act 1: Building the Pipeline (Chapters 3–4)

A complete pipeline was built: news collection → embedding/sentiment extraction → merge with market data → binary classification. Under single-window evaluation (70/15/15 walk-forward split), the Transformer with FinBERT-PT-BR sentiment features achieves **ROC-AUC = 0.709** on ITUB4 (h=21 days). This appears to demonstrate that domain-specific sentiment outperforms generic embeddings.

### Act 2: The Self-Correction (Chapter 5)

Systematic investigation with **1,500+ model runs** reveals that the 0.709 result is an artifact:

- The Transformer exhibits **bimodal collapse** (AUC ranges from 0.08 to 0.93 depending on random seed)
- Under **expanding-window CV**, the Transformer drops to ~0.51 (near random)
- **Ablation** shows sentiment adds Δ = +0.003 to a price-only baseline (Wilcoxon p = 0.49)
- **Power analysis** confirms this effect is 44–74× below the minimum detectable effect size

### Final Answer

Under methodologically rigorous evaluation, FinBERT-PT-BR sentiment features **do not add measurable predictive signal** to an autoregressive baseline of 5 price features, for any of the 3 stocks tested. The original AUC = 0.709 is an artifact of single-window evaluation.

## Pipeline Architecture

```
1.news/          →  News JSON (InfoMoney API)
    ↓
2.stocks/        →  Market features (Yahoo Finance) + Ollama embeddings
    ↓
3.model_traning/ →  Initial models with generic embeddings
    ↓
4.finbert-br/    →  FinBERT-PT-BR sentiment (5 features/day)
    ↓
5.threshold-tuning/ → Horizon + threshold experiments
    ↓
6.17years-news/  →  Extended history (concept drift)
    ↓
7.model-evaluation/ → Diagnostics (SHAP, calibration, temporal)
    ↓
8.multi-source-news/ → Multi-source exploration (CVM, Google News)
    ↓
9.baselines/     →  Rigorous validation (1,500+ runs)
```

## Tech Stack

- **Language**: Python 3.13
- **Deep Learning**: PyTorch (BiLSTM, Transformer, TCN)
- **Classical ML**: XGBoost, scikit-learn (Logistic Regression, Random Forest)
- **NLP**: transformers (FinBERT-PT-BR), Ollama (qwen3-embedding:4b)
- **Data**: pandas, numpy, yfinance
- **Visualization**: matplotlib
- **Statistics**: scipy (Wilcoxon), bootstrap CI
- **Thesis**: LaTeX with abnTeX2 (ABNT formatting)

## Experiment Summary

| # | Experiment | Runs | Key Finding |
|---|---|---:|---|
| 1 | Autoregressive baseline | 4 | XGB on price features: AUC = 0.658 |
| 2 | Naive baselines | 3 | Majority/coin-flip/persistence: AUC ≈ 0.50 |
| 3 | Dimensionality control | 20 | Random 5-dim Ollama: AUC = 0.509 ± 0.057 |
| 4 | Bootstrap CI (Stage 4) | 1 | Re-run with seed 42: AUC = 0.442 |
| 5 | Multi-seed ITUB4 | 40 | Transformer std = 0.261 (bimodal) |
| 6 | Multi-seed × multi-ticker | 120 | Pattern replicates across 3 stocks |
| 7 | Ensemble + backtest | 20 | Long/flat strategy: Sharpe = −1.29 |
| 8 | Expanding-window CV | 145 | Baseline 0.667 vs Transformer 0.509 |
| 9 | Horizon sweep | 6 | AUC increases with horizon (class imbalance) |
| 10 | VALE3 deep-dive | 880 | Wilcoxon p = 0.194, bimodal confirmed |
| 11 | Ablation PRICE/SENT/PRICE+SENT | 225 | Δ = +0.003, p = 0.49 |
| 12 | Power analysis | analytic | MDE = 0.132–0.221 for test sizes used |
| 13 | TCN validation | 45 | TCN drops from 0.643 to 0.556 under CV |
| — | **Total** | **~1,510** | |
