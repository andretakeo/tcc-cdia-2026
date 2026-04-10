# 06 — Threshold Tuning and Horizon Experiments (Stage 5)

## Directory: `5.threshold-tuning/`

## Purpose

Explore whether different prediction horizons (h=5 vs h=21) and decision thresholds improve model performance. This stage also engineers additional sentiment-derived features.

## Experiments

### Horizon h=5 (1 trading week)

Models were retrained with a shorter prediction horizon (5 days instead of 21). The shorter horizon increases task difficulty but may better capture the temporal footprint of news-driven price movements.

### Threshold Optimization

Instead of the default 0.5 threshold for binary classification, the optimal threshold was searched on the validation set using macro F1 as the criterion.

### Engineered Sentiment Features

Additional features were derived from the base FinBERT outputs to potentially improve signal extraction.

## Results

The TCN [32,32] with engineered features achieves AUC = 0.643 under single-window evaluation — the best "practical" result of the thesis (as opposed to the Transformer's 0.709, which is more volatile).

**This result is also later shown to be an artifact** in Stage 9's TCN validation experiment.

## Figures

| File | Description |
|---|---|
| `5.threshold-tuning/roc_h5.png` | ROC curve for h=5 models |
| `5.threshold-tuning/confusion_h5.png` | Confusion matrices for h=5 |
| `5.threshold-tuning/feature_importance_h5.png` | Feature importance for h=5 |
| `5.threshold-tuning/threshold_search.png` | Threshold search visualization |
| `5.threshold-tuning/threshold_search_h5.png` | Threshold search for h=5 |
| `5.threshold-tuning/roc_threshold_optimized.png` | ROC with optimized thresholds |
| `5.threshold-tuning/confusion_matrices_optimized.png` | Confusion matrices with optimized thresholds |

## Key Design Decision

The TCN (Temporal Convolutional Network) was introduced at this stage as an alternative to the Transformer, motivated by:
- Causal convolutions prevent information leakage by construction
- Dilated convolutions provide exponentially growing receptive field
- More parameter-efficient than attention mechanisms for the dataset size
- Architecture from Bai et al. (2018)
