# 09 — Multi-Source News Exploration (Stage 8)

## Directory: `8.multi-source-news/`

## Purpose

Explore whether diversifying news sources beyond InfoMoney (adding CVM regulatory filings and Google News) produces different or better sentiment signals. This was an exploratory stage — results are not included in the formal thesis chapters.

## Sources

### CVM (Comissão de Valores Mobiliários)
- Brazilian SEC equivalent
- Regulatory filings, material facts, corporate announcements
- Collected via `cvm_collector.py`

### Google News
- Aggregated news from multiple sources
- Collected via `google_news_collector.py` using the `gnews` library and RSS feeds
- Full article text extracted with `newspaper3k`

## Analysis

The multi-source sentiment was compared against InfoMoney-only sentiment through:

1. **Pairwise logit comparison** — Do different sources produce different sentiment for the same events?
2. **3D scatter visualization** — Sentiment space exploration across sources
3. **t-SNE embedding** — Dimensionality reduction to visualize source clustering
4. **Source distance analysis** — Quantifying divergence between sources

## Figures

| File | Description |
|---|---|
| `8.multi-source-news/results/pairwise_logits.png` | Logit comparison across sources |
| `8.multi-source-news/results/scatter3d_sentiment.png` | 3D sentiment scatter |
| `8.multi-source-news/results/source_distances.png` | Inter-source sentiment distances |
| `8.multi-source-news/results/test_pipeline_overview.png` | Multi-source pipeline overview |
| `8.multi-source-news/results/tsne_sentiment.png` | t-SNE of sentiment embeddings |

## Why This Isn't in the Thesis

The multi-source exploration was conducted before the methodological investigation (Stage 9) revealed that sentiment features themselves — regardless of source — add no measurable signal. Diversifying sources addresses a question ("is InfoMoney insufficient?") that is moot once the fundamental null result is established.

The thesis mentions multi-source diversification as a direction for future work (Section 6.4).

## Key Files

- `8.multi-source-news/cvm_collector.py` — CVM data collector
- `8.multi-source-news/google_news_collector.py` — Google News collector
- `8.multi-source-news/multi_source_daily_sentiment.csv` — Merged daily sentiment
