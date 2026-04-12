"""
Carregamento e preparacao de dados para cada configuracao de estagio.
Cada funcao retorna dados prontos para treino em formato padronizado:
  - X_seq: (N, window, features) para modelos sequenciais
  - y_seq: (N,) targets
  - dates_seq: DatetimeIndex
  - X_flat: (N, features) para modelos tabulares
  - y_flat: (N,) targets
  - dates_flat: DatetimeIndex
  - feature_names: list[str]
  - description: str
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(ROOT)


def _make_sequences(X_scaled, y_arr, dates_index, window=30):
    sequences, labels, dates = [], [], []
    for i in range(window, len(X_scaled)):
        sequences.append(X_scaled[i - window : i])
        labels.append(y_arr[i])
        dates.append(dates_index[i])

    X_seq = np.array(sequences, dtype=np.float32)
    y_out = np.array(labels, dtype=np.float32)
    dates = pd.DatetimeIndex(dates)
    return X_seq, y_out, dates


def _prepare_target_and_scale(X, horizon, train_ratio=0.7):
    close = X["Close"]
    target = (close.shift(-horizon) > close).astype(int)
    target = target.iloc[:-horizon]
    X_trimmed = X.iloc[:-horizon].copy()

    feature_names = X_trimmed.columns.tolist()

    n_train = int(len(X_trimmed) * train_ratio)
    scaler = StandardScaler()
    scaler.fit(X_trimmed.iloc[:n_train])
    X_scaled = scaler.transform(X_trimmed)

    return X_scaled, target.values, X_trimmed.index, feature_names


def _normalize_index(idx):
    """Remove timezone info if present."""
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def load_stage3_ollama(horizon=21, pca_components=32, window=30):
    csv_path = os.path.join(PROJECT_ROOT, "2.stocks", "dataset_full.csv")
    X_full = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    X_full.index = _normalize_index(pd.to_datetime(X_full.index, utc=True))

    # Separate embeddings and price
    emb_cols = [c for c in X_full.columns if c.startswith("emb_")]
    price_cols = [c for c in X_full.columns if not c.startswith("emb_")]

    # Target
    close = X_full["Close"]
    target = (close.shift(-horizon) > close).astype(int)
    target = target.iloc[:-horizon]
    X = X_full.iloc[:-horizon].copy()

    # PCA on embeddings
    if emb_cols:
        pca = PCA(n_components=pca_components, random_state=42)
        emb_reduced = pca.fit_transform(X[emb_cols])
        emb_df = pd.DataFrame(
            emb_reduced, index=X.index,
            columns=[f"pca_{i}" for i in range(pca_components)],
        )
        X = pd.concat([X[price_cols], emb_df], axis=1)

    feature_names = X.columns.tolist()

    # Normalize (fit on train only)
    n_train = int(len(X) * 0.7)
    scaler = StandardScaler()
    scaler.fit(X.iloc[:n_train])
    X_scaled = scaler.transform(X)

    # Sequences
    X_seq, y_seq, dates_seq = _make_sequences(X_scaled, target.values, X.index, window)

    return {
        "X_seq": X_seq, "y_seq": y_seq, "dates_seq": dates_seq,
        "X_flat": X_scaled, "y_flat": target.values, "dates_flat": X.index,
        "feature_names": feature_names,
        "description": "Stage 3: Ollama embeddings (1024 -> PCA 32) + preco",
    }


def load_stage4_finbert_4y(horizon=21, window=30):
    sentiment_path = os.path.join(PROJECT_ROOT, "4.finbert-br", "itub4_daily_sentiment.csv")
    sentiment = pd.read_csv(sentiment_path, parse_dates=["date"], index_col="date")

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "6.17years-news"))
    from yahoo_finance import MarketData
    prices = MarketData("ITUB4.SA").features(lags=5)
    prices.index = _normalize_index(prices.index)

    X_full = prices.join(sentiment, how="left").ffill().dropna()

    X_scaled, y, dates, feature_names = _prepare_target_and_scale(X_full, horizon)
    X_seq, y_seq, dates_seq = _make_sequences(X_scaled, y, dates, window)

    return {
        "X_seq": X_seq, "y_seq": y_seq, "dates_seq": dates_seq,
        "X_flat": X_scaled, "y_flat": y, "dates_flat": dates,
        "feature_names": feature_names,
        "description": "Stage 4: FinBERT sentiment + preco (4 anos)",
    }


def load_stage5_horizon5(window=30):
    # Load prices (no embeddings)
    csv_path = os.path.join(PROJECT_ROOT, "2.stocks", "dataset_full.csv")
    X_full = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    X_full.index = _normalize_index(pd.to_datetime(X_full.index, utc=True))
    price_cols = [c for c in X_full.columns if not c.startswith("emb_")]
    X_prices = X_full[price_cols].copy()

    # Load sentiment
    sentiment_path = os.path.join(PROJECT_ROOT, "4.finbert-br", "itub4_daily_sentiment.csv")
    sentiment = pd.read_csv(sentiment_path, parse_dates=["date"], index_col="date")

    X_combined = X_prices.join(sentiment, how="left")
    sent_cols = sentiment.columns.tolist()
    X_combined[sent_cols] = X_combined[sent_cols].ffill()

    # Feature engineering
    for col in ['mean_logit_pos', 'mean_logit_neg', 'mean_logit_neu', 'mean_sentiment']:
        X_combined[f'{col}_ma7'] = X_combined[col].rolling(7).mean()
        X_combined[f'{col}_ma21'] = X_combined[col].rolling(21).mean()

    for col in ['mean_logit_pos', 'mean_logit_neg', 'mean_sentiment']:
        X_combined[f'{col}_delta7'] = X_combined[col] - X_combined[f'{col}_ma7']

    X_combined['pos_neg_ratio'] = X_combined['mean_logit_pos'] / (X_combined['mean_logit_neg'].abs() + 1e-6)
    X_combined['pos_neg_ratio_ma7'] = X_combined['pos_neg_ratio'].rolling(7).mean()
    X_combined['n_articles_sum7'] = X_combined['n_articles'].rolling(7).sum()
    X_combined['n_articles_sum21'] = X_combined['n_articles'].rolling(21).sum()
    X_combined['sentiment_std7'] = X_combined['mean_sentiment'].rolling(7).std()
    X_combined = X_combined.dropna()

    horizon = 5
    X_scaled, y, dates, feature_names = _prepare_target_and_scale(X_combined, horizon)
    X_seq, y_seq, dates_seq = _make_sequences(X_scaled, y, dates, window)

    return {
        "X_seq": X_seq, "y_seq": y_seq, "dates_seq": dates_seq,
        "X_flat": X_scaled, "y_flat": y, "dates_flat": dates,
        "feature_names": feature_names,
        "description": "Stage 5b: FinBERT + features engenheiradas, horizonte 5 dias",
    }


def load_stage6_finbert_17y(horizon=21, window=30):
    sentiment_path = os.path.join(PROJECT_ROOT, "6.17years-news", "itub4_daily_sentiment_17y.csv")
    sentiment = pd.read_csv(sentiment_path, parse_dates=["date"], index_col="date")

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "6.17years-news"))
    from yahoo_finance import MarketData
    prices = MarketData("ITUB4.SA").features(lags=5)
    prices.index = _normalize_index(prices.index)

    X_full = prices.join(sentiment, how="left").ffill().dropna()

    X_scaled, y, dates, feature_names = _prepare_target_and_scale(X_full, horizon)
    X_seq, y_seq, dates_seq = _make_sequences(X_scaled, y, dates, window)

    return {
        "X_seq": X_seq, "y_seq": y_seq, "dates_seq": dates_seq,
        "X_flat": X_scaled, "y_flat": y, "dates_flat": dates,
        "feature_names": feature_names,
        "description": "Stage 6: FinBERT sentiment + preco (17 anos)",
    }
