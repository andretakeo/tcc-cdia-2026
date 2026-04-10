# 08 — Model Diagnostics (Stage 7: Comprehensive Evaluation)

## Directory: `7.model-evaluation/`

## Purpose

Apply comprehensive diagnostic tools to all models across all pipeline stages (3, 4, 5, 6) to understand *why* models behave as they do, not just *how well* they perform.

## Shared Model Library: `7.model-evaluation/shared/models.py`

This file contains standardized implementations of all 6 model architectures:

### PyTorch Models

| Model | Architecture | Key Parameters |
|---|---|---|
| `LSTMClassifier` | 2-layer BiLSTM | hidden=128, dropout=0.3, sigmoid output |
| `TransformerClassifier` | Encoder + positional encoding + MLP head | 2 layers, 4 heads, d_model=64, dropout=0.2 |
| `TCNClassifier` | Causal dilated convolutions + residual blocks | [32,32] channels, kernel_size=3, sigmoid output |

### Scikit-learn / XGBoost Models

| Model | Key Parameters |
|---|---|
| `LogisticRegression` | class_weight='balanced', solver='lbfgs', C=1.0 |
| `RandomForestClassifier` | 200 estimators, balanced weights, max_features='sqrt' |
| `XGBClassifier` | 200 estimators, max_depth=4, scale_pos_weight=imbalance ratio |

**Note**: The TCN uses `Sigmoid` output → must use `nn.BCELoss`, not `BCEWithLogitsLoss`.

## Diagnostic Suite

For each (model, stage) combination, the following diagnostics were generated:

### 1. ROC Curves (`stage{N}_roc.png`)
Standard ROC curve with AUC annotation for all models.

### 2. Confusion Matrices (`stage{N}_confusion.png`)
Normalized confusion matrices revealing prediction distribution. Critical for detecting degenerate classifiers (e.g., Transformer predicting "Up" 94% of the time).

### 3. Prediction Distributions (`stage{N}_distributions.png`)
Histogram of predicted probabilities by true class. Healthy classifiers show separated distributions; degenerate ones show overlapping or collapsed distributions.

### 4. Calibration Curves (`stage{N}_calibration.png`)
Reliability diagrams comparing predicted probability to observed frequency. Reports Expected Calibration Error (ECE) and Brier Score.

### 5. Learning Curves (`stage{N}_learning_curves.png`)
Training and validation loss/metrics as a function of training set size. Reveals underfitting vs overfitting regimes.

### 6. SHAP Analysis (`stage{N}_shap_{model}.png`)
SHapley Additive Explanations for tree-based models (XGBoost, Random Forest, Logistic Regression), showing which features drive predictions.

### 7. Permutation Importance (`stage{N}_perm_{model}.png`)
Feature importance via random permutation, complementing SHAP with a model-agnostic approach.

### 8. Temporal Stability (`stage{N}_temporal.png`)
Rolling 3-month window AUC to assess performance stability over time.

## Key Findings from Diagnostics

1. **SHAP**: Price-derived features (return, lags, volatility) consistently dominate sentiment features in importance rankings
2. **Calibration**: All models show poor calibration (ECE > 0.1), consistent with the low-data regime
3. **Temporal stability**: Performance fluctuates substantially across 3-month windows, confirming non-stationarity
4. **Prediction distributions**: Transformer distributions are heavily collapsed toward one class

## Complete Figure Inventory (48 diagnostic plots)

Stages 3, 4, 5, 6 × 12 plot types each:

```
7.model-evaluation/results/
├── stage3_calibration.png          ├── stage4_calibration.png
├── stage3_confusion.png            ├── stage4_confusion.png
├── stage3_distributions.png        ├── stage4_distributions.png
├── stage3_learning_curves.png      ├── stage4_learning_curves.png
├── stage3_perm_logistic_regression.png   ...
├── stage3_perm_random_forest.png
├── stage3_perm_xgboost.png
├── stage3_roc.png
├── stage3_shap_logistic_regression.png
├── stage3_shap_random_forest.png
├── stage3_shap_xgboost.png
├── stage3_temporal.png
└── (same pattern for stage4, stage5, stage6)
```

## Supporting Modules

| File | Purpose |
|---|---|
| `shared/models.py` | All model architectures |
| `shared/data_loader.py` | Data loading for 4 stages |
| `shared/metrics.py` | Comprehensive metric calculations |
| `shared/trainer.py` | Training orchestration |
| `shared/plots.py` | Visualization functions |
