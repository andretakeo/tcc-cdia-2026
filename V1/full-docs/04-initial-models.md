# 04 — Initial Models with Generic Embeddings (Stage 3)

## Directory: `3.model_traning/`

## Purpose

Train binary classifiers on the combined price + Ollama embedding features to establish an initial performance baseline. This stage tests whether generic high-dimensional embeddings contain predictive signal for stock direction.

## Target Variable

- **Binary**: `1 if Close[t+21] > Close[t]`, else `0`
- **Class balance**: ~59% "Up" / 41% "Down"
- **Horizon**: 21 trading days (~1 month)

## Feature Preprocessing

- Embeddings reduced from 1,024 → 32 dimensions via PCA
- **Known issue**: PCA fitted on full dataset (train+val+test), introducing variance leakage
- Final feature count: 11 (price) + 32 (PCA embeddings) = 43 features

## Evaluation Protocol

- Walk-forward split: 70% train / 15% validation / 15% test (chronological, no shuffle)
- Single seed (42)
- No confidence intervals reported at this stage

## Models and Results

| Model | Architecture | ROC-AUC |
|---|---|---:|
| BiLSTM Original | 2 layers, 128 hidden, 30% dropout | 0.443 |
| BiLSTM Reduced | 1 layer, 32 hidden, 50% dropout | 0.505 |
| Transformer | 2 layers, 4 heads, d_model=64 | 0.568 |
| **XGBoost** | 300 trees, max_depth=4 | **0.610** |

## Interpretation

All models perform near chance level, suggesting that generic 1,024-dim embeddings do not contain useful predictive signal for this task. XGBoost performs best, likely because its tree-based structure handles the noise in high-dimensional features better than neural architectures in the low-data regime (~800 training samples).

**This motivated the hypothesis tested in Stage 4**: perhaps a compact, domain-specific representation (FinBERT sentiment) would be more informative.

## Figures

| File | Description |
|---|---|
| `3.model_traning/lstm_results.png` | BiLSTM training curves |
| `3.model_traning/transformer_results.png` | Transformer training curves |
| `3.model_traning/xgboost_roc.png` | XGBoost ROC curve |
| `3.model_traning/roc_comparison.png` | All models ROC comparison |

## Known Bug: `pos_weight`

In `lstm_classifier.py` (line 256–257), `pos_weight` is computed as `(1 - y_mean) / y_mean ≈ 0.69`, which *reduces* minority class weight instead of increasing it. Additionally, the code creates `BCEWithLogitsLoss(pos_weight=...)` but then uses `nn.BCELoss()` in the training loop (line 276), so the weight is never applied. This bug affects BiLSTM results in Stages 3 and 4 but not Stage 9 experiments, which use corrected implementations.

## Key Files

- `3.model_traning/lstm_classifier.py` — BiLSTM implementation + training
- `3.model_traning/transformer_classifier.py` — Transformer implementation + training
- `3.model_traning/xgboost_baseline.py` — XGBoost baseline
