# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TCC (undergraduate thesis) investigating whether Brazilian financial news improves stock price direction prediction for B3 equities (ITUB4, PETR4, VALE3). The thesis concludes with a methodological self-correction: the initial positive result (AUC 0.709) is shown to be a single-window evaluation artifact through 1,435+ model runs.

## Architecture

Sequential data pipeline organized by numbered directories:

- **1.news/** — News collection from InfoMoney WordPress API (`ExtratorDeNoticias` class)
- **2.stocks/** — Feature engineering: Yahoo Finance OHLCV (`MarketData`) + Ollama embeddings (`NewsEmbedder`) → merged dataset
- **3.model_traning/** — Initial model training with generic embeddings (BiLSTM, Transformer, XGBoost)
- **4.finbert-br/** — FinBERT-PT-BR sentiment extraction (5 features per day)
- **5.threshold-tuning/** — Threshold + horizon experiments, engineered sentiment features
- **6.17years-news/** — Extended historical data (concept drift study)
- **7.model-evaluation/** — Diagnostic evaluation (SHAP, calibration, temporal stability). Shared models in `7.model-evaluation/shared/models.py`
- **8.multi-source-news/** — Multi-source news collection (CVM, Google News)
- **9.baselines/** — Rigorous validation: multi-seed, expanding-window CV, ablation, power analysis. Shared utilities in `9.baselines/eval_utils.py`

Data flows: News JSON → embeddings/sentiment → merge with price features (left join + forward-fill) → windowed sequences (30 days) → binary classification (sobe/desce)

## Running Code

Notebooks are the primary execution units. No build system, Makefile, or test framework exists.

```bash
# Run a notebook
cd 9.baselines && jupyter nbconvert --execute notebook.ipynb --to notebook --inplace

# Compile the thesis (requires MiKTeX or TeX Live with abnTeX2)
cd docs && pdflatex tcc.tex && bibtex tcc && pdflatex tcc.tex && pdflatex tcc.tex
```

## Key Shared Modules

- **`9.baselines/eval_utils.py`** — `walk_forward_split()`, `bootstrap_auc_ci()`, `evaluate_model()`, `make_binary_target()`. Used by all Chapter 5 experiments.
- **`7.model-evaluation/shared/models.py`** — `TCNClassifier`, `TransformerClassifier`, `LSTMClassifier`, `build_xgboost()`, `build_random_forest()`. All PyTorch models output sigmoid probabilities.
- **`2.stocks/yahoo_finance.py`** — `MarketData` class wrapping yfinance
- **`2.stocks/news_embedder.py`** — `NewsEmbedder` class wrapping Ollama embeddings
- **`1.news/extractor.py`** — `ExtratorDeNoticias` class for InfoMoney API

## Tech Stack

Python 3.13, PyTorch, XGBoost, scikit-learn, pandas, numpy, yfinance, transformers (FinBERT), Ollama, matplotlib, scipy. No requirements.txt — dependencies are installed manually.

## Conventions

- **Language**: Code in English (snake_case), domain terms in Portuguese (e.g., `Artigo`, `ExtratorDeNoticias`, variable names like `acao`, `artigos`)
- **Type hints**: Python 3.10+ syntax (`str | None`)
- **Thesis text**: All in pt-br (docs/capitulo_4.md, docs/capitulo_5.md, docs/tcc.tex)
- **ABNT formatting**: LaTeX uses `abnTeX2` class with `abntex2cite` (estilo alf)
- **Target binary**: `1 if Close[t+h] > Close[t]` with h=5 or h=21 days
- **Walk-forward split**: 70% train / 15% val / 15% test, chronological, no shuffle
- **Bootstrap CI**: Always 1,000 resamples via `bootstrap_auc_ci()` when reporting AUC
- **Seed management**: `torch.manual_seed(seed)` + `np.random.seed(seed)` at experiment start

## Important Context

- The README.md still reflects the initial positive narrative (AUC 0.709). The corrected conclusion is in docs/capitulo_5.md and docs/tcc.tex.
- The "dumb baseline" was renamed to "baseline autoregressivo" throughout the thesis text.
- Forward-fill of sentiment on non-news days is a known look-ahead risk, discussed in docs/capitulo_4.md Section 4.3.
- TCNClassifier uses Sigmoid output → use `nn.BCELoss`, not `BCEWithLogitsLoss`.
