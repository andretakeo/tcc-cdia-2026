# 03 — Feature Engineering (Stage 2: Stocks + Embeddings)

## Directory: `2.stocks/`

## Purpose

Combine market data (price/volume features) with textual representations (Ollama embeddings) into a single dataset for model training.

## Market Data: `yahoo_finance.py`

The `MarketData` class wraps `yfinance` to obtain daily OHLCV data and compute 11 technical features per day:

| # | Feature | Description |
|---|---|---|
| 1 | `Close` | Closing price |
| 2 | `Volume` | Trading volume |
| 3 | `return` | Daily return: `(Close[t] - Close[t-1]) / Close[t-1]` |
| 4 | `ma7` | 7-day moving average |
| 5 | `ma21` | 21-day moving average |
| 6 | `std21` | 21-day rolling standard deviation (volatility) |
| 7 | `lag_1` | Close price lagged 1 day |
| 8 | `lag_2` | Close price lagged 2 days |
| 9 | `lag_3` | Close price lagged 3 days |
| 10 | `lag_4` | Close price lagged 4 days |
| 11 | `lag_5` | Close price lagged 5 days |

## Textual Embeddings: `news_embedder.py`

The `NewsEmbedder` class generates dense vector representations of news articles:

1. **Model**: Ollama `qwen3-embedding:4b` (local inference)
2. **Input**: Article title + excerpt
3. **Output**: 1,024-dimensional embedding vector per article
4. **Aggregation**: When multiple articles exist for the same day, embeddings are combined via **recency-weighted mean** (more recent articles receive higher weight)

## Dataset Merge

The merge follows this protocol:

1. **Left join** on date: market data (daily) joined with news embeddings (sparse)
2. **Forward-fill**: Days without news carry forward the most recent embedding
3. **Result**: 1,227 days × 1,035 features (11 price + 1,024 embedding)

## Design Decisions

| Decision | Rationale |
|---|---|
| 11 price features | Standard technical analysis features covering price, volume, momentum, and volatility |
| Ollama embeddings (1,024-dim) | Local inference, no API costs, moderate-quality generic embeddings |
| Recency-weighted aggregation | More recent articles are presumably more relevant |
| Forward-fill for missing days | Preserves temporal causality (only past information propagates forward) |
| No intraday alignment | Simplicity; risk is mitigated by multi-day horizons |

## Known Issues

1. **PCA leakage (Stage 3 only)**: PCA reduction from 1,024 → 32 dimensions was fitted on the entire dataset (train + val + test), introducing variance structure leakage. This does NOT affect Stage 4+ results, which use FinBERT features without PCA.

2. **Forward-fill look-ahead risk**: Articles published after B3 close (~17:00 BRT) are assigned to day `t` and aligned with day `t`'s price data. The impact is mitigated by 5–21 day horizons (see `docs/capitulo_4.md` Section 4.3 for detailed analysis).

## Output

- `2.stocks/dataset_full.csv` — Complete merged dataset

## Key Files

- `2.stocks/yahoo_finance.py` — `MarketData` class
- `2.stocks/news_embedder.py` — `NewsEmbedder` class
