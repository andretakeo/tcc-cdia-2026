# 02 — Data Collection (Stage 1: News)

## Directory: `1.news/`

## Purpose

Collect a corpus of Brazilian financial news articles to serve as the textual input for sentiment extraction and embedding generation.

## Source: InfoMoney

InfoMoney was chosen as the sole news source for the following reasons:

- One of Brazil's largest financial news portals
- Focuses on retail investors (B3-oriented content)
- Exposes a public WordPress REST API (`/wp-json/wp/v2/posts`) that allows programmatic access
- Articles contain structured metadata (title, excerpt, date, categories)

**Trade-off acknowledged**: Single-source bias limits generalizability. The InfoMoney editorial lens is oriented toward retail investors, not institutional analysis. This is discussed in the thesis Limitations section.

## Implementation: `extractor.py`

The `ExtratorDeNoticias` class handles:

1. **Paginated search**: Queries the WordPress API with 100 articles per page per ticker
2. **Retry logic**: Up to 3 attempts with linear backoff (1s, 2s) for timeouts, connection errors, and 429/5xx status codes
3. **Preprocessing**: Removes HTML tags, decodes entities, normalizes whitespace
4. **Parallelism**: Uses `ThreadPoolExecutor` for concurrent extraction across tickers

## Data Volume

| Ticker | Articles | Period |
|---|---:|---|
| ITUB4 (Itaú Unibanco) | 2,572 | 2009–2026 |
| PETR4 (Petrobras) | 1,775 | 2009–2026 |
| VALE3 (Vale) | 1,525 | 2009–2026 |
| **Total** | **5,872** | |

## Output Format

Each ticker produces a JSON file (`{ticker}_noticias.json`) with articles containing:
- `title`: Article headline
- `excerpt`: Summary/lead paragraph
- `date`: ISO 8601 timestamp (e.g., `2026-03-16T10:56:56`)
- `content`: Full article text (HTML cleaned)

## Design Decisions

| Decision | Rationale |
|---|---|
| WordPress API over web scraping | Structured data, no fragile HTML parsing, respects robots.txt |
| Title + excerpt as primary input | More concise and focused than full body text; fits BERT's 512-token limit |
| No intraday timestamp filtering | Acknowledged as look-ahead risk; mitigated by 5–21 day horizons |
| Single source (InfoMoney) | Simplicity and API availability; multi-source explored in Stage 8 |

## Key File

- `1.news/extractor.py` — The `ExtratorDeNoticias` class
