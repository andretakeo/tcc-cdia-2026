# 11 — Complete Figures Catalog

## Stage 3: Initial Models (`3.model_traning/`)

| File | Description | Used in Thesis |
|---|---|---|
| `lstm_results.png` | BiLSTM training loss/accuracy curves | No (supporting) |
| `transformer_results.png` | Transformer training curves | No (supporting) |
| `xgboost_roc.png` | XGBoost ROC curve on test set | No (supporting) |
| `roc_comparison.png` | All 3 models ROC comparison | No (supporting) |

## Stage 4: FinBERT Sentiment (`4.finbert-br/`)

| File | Description | Used in Thesis |
|---|---|---|
| `lstm_results.png` | BiLSTM training curves with sentiment | No |
| `transformer_results_finbert.png` | Transformer training curves with sentiment | No |
| `xgboost_roc_finbert.png` | XGBoost ROC with sentiment features | No |
| `roc_comparison_finbert.png` | All models ROC comparison (Stage 4) | No |

## Stage 5: Threshold Tuning (`5.threshold-tuning/`)

| File | Description | Used in Thesis |
|---|---|---|
| `roc_h5.png` | ROC curves for horizon h=5 | No |
| `confusion_h5.png` | Confusion matrices for h=5 | No |
| `feature_importance_h5.png` | Feature importance ranking for h=5 | No |
| `threshold_search.png` | Decision threshold search | No |
| `threshold_search_h5.png` | Threshold search for h=5 | No |
| `roc_threshold_optimized.png` | ROC with optimized thresholds | No |
| `confusion_matrices_optimized.png` | Confusion matrices after optimization | No |

## Stage 6: 17-Year History (`6.17years-news/`)

| File | Description | Used in Thesis |
|---|---|---|
| `lstm_results.png` | BiLSTM with 17-year data | No |
| `roc_17years.png` | ROC curves with extended data | No |
| `transformer_17y.png` | Transformer with extended data | No |
| `xgboost_roc_17y.png` | XGBoost with extended data | No |

## Stage 7: Model Diagnostics (`7.model-evaluation/results/`)

48 figures total — 12 per stage × 4 stages. Pattern: `stage{N}_{diagnostic}.png`

| Diagnostic | Description | Stages |
|---|---|---|
| `calibration` | Reliability diagram, ECE, Brier score | 3, 4, 5, 6 |
| `confusion` | Normalized confusion matrices | 3, 4, 5, 6 |
| `distributions` | Predicted probability histograms by class | 3, 4, 5, 6 |
| `learning_curves` | Train/val metrics vs training size | 3, 4, 5, 6 |
| `roc` | ROC curves for all models | 3, 4, 5, 6 |
| `temporal` | Rolling 3-month AUC stability | 3, 4, 5, 6 |
| `shap_logistic_regression` | SHAP feature importance | 3, 4, 5, 6 |
| `shap_random_forest` | SHAP feature importance | 3, 4, 5, 6 |
| `shap_xgboost` | SHAP feature importance | 3, 4, 5, 6 |
| `perm_logistic_regression` | Permutation importance | 3, 4, 5, 6 |
| `perm_random_forest` | Permutation importance | 3, 4, 5, 6 |
| `perm_xgboost` | Permutation importance | 3, 4, 5, 6 |

## Stage 8: Multi-Source News (`8.multi-source-news/results/`)

| File | Description | Used in Thesis |
|---|---|---|
| `pairwise_logits.png` | Logit comparison across news sources | No |
| `scatter3d_sentiment.png` | 3D sentiment space visualization | No |
| `source_distances.png` | Inter-source sentiment distances | No |
| `test_pipeline_overview.png` | Multi-source pipeline diagram | No |
| `tsne_sentiment.png` | t-SNE of sentiment embeddings | No |

## Stage 9: Rigorous Validation (`9.baselines/`)

**These are the thesis figures (Chapter 5):**

| File | Description | Thesis Figure | Thesis Reference |
|---|---|---|---|
| `multi_seed_histograms.png` | AUC distribution over 20 seeds (bimodal vs stable) | Figure 1 | `\ref{fig:multiseed}` |
| `multi_seed_tradeoff.png` | AUC vs number of "Down" predictions | Supporting | — |
| `multi_seed_multi_ticker.png` | Multi-ticker seed distributions | Supporting | — |
| `expanding_cv_overtime.png` | AUC by fold over time, 3 tickers | Figure 2 | `\ref{fig:expandingcv}` |
| `expanding_cv_hero.png` | AUC vs class-prior shift scatter | Supporting | — |
| `ensemble_backtest.png` | Equity curves: ensemble vs buy-and-hold | Supporting | — |
| `vale3_deepdive.png` | VALE3 AUC over folds | Supporting | — |
| `vale3_deepdive_hist.png` | 880-run AUC distribution (bimodal) | Figure 3 | `\ref{fig:vale3}` |
| `ablation_boxplot.png` | PRICE vs SENT vs PRICE+SENT (h=21) | Figure 4 | `\ref{fig:ablation}` |
| `ablation_h5_boxplot.png` | PRICE vs SENT vs PRICE+SENT (h=5) | Supporting | — |
| `horizon_sweep.png` | AUC + CI vs prediction horizon | Supporting | — |
| `power_analysis.png` | MDE vs observed effects | Supporting | — |
| `tcn_validation_hist.png` | TCN multi-seed + CV distributions | Figure 5 | `\ref{fig:tcn}` |
| `tcn_validation_shift.png` | TCN AUC vs class-prior shift | Supporting | — |
| `ffill_sensitivity_plot.png` | Same-day vs lag-1d comparison | Supporting | — |

## Total Figure Count

| Location | Count |
|---|---:|
| Stage 3 | 4 |
| Stage 4 | 4 |
| Stage 5 | 7 |
| Stage 6 | 4 |
| Stage 7 | 48 |
| Stage 8 | 5 |
| Stage 9 | 15 |
| **Total** | **87** |

Figures used in thesis LaTeX (`tcc.tex`): **5** (Figures 1–5 from Stage 9)
