import yfinance as yf
import pandas as pd


class MarketData:
    """Busca dados de mercado do Yahoo Finance prontos para modelagem."""

    def __init__(self, tickers: str | list[str]):
        self.tickers = [tickers] if isinstance(tickers, str) else tickers

    # ── Dados brutos ────────────────────────────────────────────────────────

    def history(
        self,
        period: str = "1y",
        interval: str = "1d",
        ticker: str | None = None,
    ) -> pd.DataFrame:
        """
        OHLCV de um ticker. Retorna DataFrame indexado por data.

        period  : 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
        interval: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
        """
        t = ticker or self.tickers[0]
        df = yf.download(t, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        return df

    def close(
        self,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fechamentos ajustados de todos os tickers em colunas."""
        df = yf.download(
            self.tickers, period=period, interval=interval,
            progress=False, auto_adjust=True,
        )["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(name=self.tickers[0])
        df.index.name = "Date"
        return df

    def info(self, ticker: str | None = None) -> dict:
        """Metadados e fundamentals brutos do ticker."""
        t = ticker or self.tickers[0]
        return yf.Ticker(t).info

    # ── Pronto para modelo ───────────────────────────────────────────────────

    def features(
        self,
        period: str = "1y",
        lags: int = 5,
        ticker: str | None = None,
    ) -> pd.DataFrame:
        """
        DataFrame com features técnicas comuns para modelos de previsão:
          - Close, Volume
          - Retorno diário
          - Médias móveis (7, 21 dias)
          - Desvio padrão 21 dias (volatilidade)
          - Lags do fechamento (lag_1 … lag_N)
        """
        df = self.history(period=period, ticker=ticker)[["Close", "Volume"]].copy()

        df["return"]   = df["Close"].pct_change()
        df["ma7"]      = df["Close"].rolling(7).mean()
        df["ma21"]     = df["Close"].rolling(21).mean()
        df["std21"]    = df["Close"].rolling(21).std()

        for i in range(1, lags + 1):
            df[f"lag_{i}"] = df["Close"].shift(i)

        return df.dropna()

    def target(
        self,
        period: str = "1y",
        horizon: int = 1,
        ticker: str | None = None,
    ) -> pd.Series:
        """
        Série com o fechamento deslocado `horizon` dias à frente —
        variável-alvo para regressão/classificação.
        """
        close = self.history(period=period, ticker=ticker)["Close"]
        return close.shift(-horizon).dropna().rename("target")