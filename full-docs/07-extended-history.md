# 07 — Extended Historical Data (Stage 6: 17-Year News)

## Directory: `6.17years-news/`

## Purpose

Test whether extending the training data from ~4 years to 17 years of news history improves model performance, and investigate concept drift in sentiment features over longer time horizons.

## Approach

- Extended InfoMoney news collection back to 2009
- Processed sentiment with FinBERT-PT-BR for the full 17-year period
- Retrained models on the larger dataset

## Results

The extended history did not materially improve results, suggesting either:
1. Concept drift degrades the value of older training data
2. The sentiment features lack predictive power regardless of data volume
3. The relationship between news sentiment and price direction is non-stationary

Stage 9's experiments later confirmed explanation (2) as the primary factor.

## Figures

| File | Description |
|---|---|
| `6.17years-news/lstm_results.png` | BiLSTM with 17-year data |
| `6.17years-news/roc_17years.png` | ROC curves with extended data |
| `6.17years-news/transformer_17y.png` | Transformer with extended data |
| `6.17years-news/xgboost_roc_17y.png` | XGBoost with extended data |

## Key Files

- `6.17years-news/yahoo_finance.py` — Extended market data wrapper
- `6.17years-news/itub4_daily_sentiment_17y.csv` — 17-year daily sentiment for ITUB4
