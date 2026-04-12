# 05 — FinBERT-PT-BR Sentiment Extraction (Stage 4)

## Directory: `4.finbert-br/`

## Purpose

Replace the 1,024-dim generic embeddings from Stage 3 with 5 compact, domain-specific sentiment features extracted by FinBERT-PT-BR — a BERT model fine-tuned on Brazilian Portuguese financial text.

## Hypothesis

The poor performance of generic embeddings (Stage 3) is due to semantic noise irrelevant to the financial domain. A compact representation informed by domain knowledge should produce better features even with far fewer dimensions (5 vs 1,024).

## The FinBERT-PT-BR Model

- **Base**: BERT architecture
- **Training**: Pre-trained on Brazilian Portuguese financial corpora (news, reports)
- **Output**: 3 logits per input — POSITIVE, NEGATIVE, NEUTRAL sentiment
- **Loading**: Local via `transformers.AutoModelForSequenceClassification` from `4.finbert-br/FinBERT-PT-BR/`

## Extraction Pipeline

1. **Input**: Title + excerpt per article, truncated at 512 tokens
2. **Batch inference**: 32 articles per batch
3. **Per-article output**: Predicted class (POS/NEG/NEU) + raw logits `[pos, neg, neu]`
4. **Persistence**: `{ticker}_noticias_sentiment.json`
5. **Daily aggregation** into 5 features:

| # | Feature | Description |
|---|---|---|
| 1 | `n_articles` | Number of articles published that day |
| 2 | `mean_logit_pos` | Mean positive logit across day's articles |
| 3 | `mean_logit_neg` | Mean negative logit across day's articles |
| 4 | `mean_logit_neu` | Mean neutral logit across day's articles |
| 5 | `mean_sentiment` | Mean ordinal sentiment (0=NEG, 1=NEU, 2=POS) |

6. **Temporal join**: Left join with 11 price features, forward-fill on non-news days
7. **Final dataset**: 1,207 days × 16 features (11 price + 5 sentiment)

## Results Under Single-Window Evaluation

| Model | AUC Stage 3 | AUC Stage 4 | Δ |
|---|---:|---:|---:|
| BiLSTM Original | 0.443 | 0.500 | +0.057 |
| BiLSTM Reduced | 0.505 | 0.477 | −0.028 |
| XGBoost | 0.610 | 0.670 | +0.060 |
| **Transformer** | 0.568 | **0.709** | **+0.141** |

**The Transformer achieves AUC = 0.709** — the strongest result of the thesis at this point.

## Warning Signs (Motivating Chapter 5)

Three observations suggested caution:

1. **Degenerate confusion matrix**: The Transformer predicts "Down" only 11 times in 177 test samples. Precision(Down) = 1.00 but Recall(Down) = 0.20. The model is essentially a majority-class predictor.

2. **Accuracy ≈ class prior**: 76.3% accuracy vs ~69% majority class in the test set. The net contribution over a constant predictor is only ~7 percentage points.

3. **No confidence intervals or multiple seeds**: The result is a single point estimate from a single random initialization.

## Forward-Fill Risk Analysis

Articles published after B3 market close (~17:00 BRT) are assigned to day `t`, creating a potential 1-day look-ahead bias. Three mitigating factors:

1. Horizons of 5–21 days make a 1-day shift <5% of the prediction window
2. Daily mean aggregation dilutes individual post-close articles
3. Forward-fill propagates past information forward (causally correct direction)

A formal sensitivity analysis in Stage 9 (`ffill_sensitivity.ipynb`) quantifies this: the bias is ~0.022 AUC points, irrelevant to the null result.

## Figures

| File | Description |
|---|---|
| `4.finbert-br/lstm_results.png` | BiLSTM training curves with sentiment features |
| `4.finbert-br/transformer_results_finbert.png` | Transformer training curves |
| `4.finbert-br/xgboost_roc_finbert.png` | XGBoost ROC curve |
| `4.finbert-br/roc_comparison_finbert.png` | All models ROC comparison |

## Key Files

- `4.finbert-br/index.ipynb` — Sentiment extraction pipeline
- `4.finbert-br/model_training.ipynb` — Model retraining with sentiment features
- `4.finbert-br/{itub4,petr4,vale3}_daily_sentiment.csv` — Daily sentiment features
- `4.finbert-br/{itub4,petr4,vale3}_noticias_sentiment.json` — Per-article sentiment
