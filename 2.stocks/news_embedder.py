"""
NewsEmbedder
─────────────────────────────────────────────────────────────────────────────
Transforma notícias do InfoMoney em features de embedding diárias,
prontas para serem concatenadas com MarketData.features() e treinar
um modelo de previsão de séries temporais.

Dependências:
    pip install ollama numpy pandas

Ollama rodando local:
    ollama pull qwen3-embedding:4b
    ollama serve
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import ollama

# ─────────────────────────────────────────────────────────────────────────────
# Logger — aparece no output do notebook com timestamp
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NewsEmbedder")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_text(article: dict, fields: list[str], max_chars: int = 50_000) -> str:
    parts = [str(article.get(f, "")).strip() for f in fields if article.get(f)]
    text = " | ".join(parts)
    return text[:max_chars]


def _parse_date(value) -> datetime:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().replace(tzinfo=None)
    dt = datetime.fromisoformat(str(value))
    return dt.replace(tzinfo=None)


def _fmt(seconds: float) -> str:
    """Formata segundos em string legível: 1m23s ou 45.2s."""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    return f"{seconds:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Classe principal
# ─────────────────────────────────────────────────────────────────────────────

class NewsEmbedder:
    """
    Embeda notícias via Ollama e agrega por dia com média ponderada
    (notícias mais recentes no dia têm peso maior).
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:4b",
        fields: list[str] = ["title", "excerpt", "content"],
        ollama_host: str = "http://localhost:11434",
        cache_path: str | None = "embeddings_cache.npz",
        summarizer_model: str | None = None,
        max_chars: int = 50_000,
    ):
        self.model = model
        self.fields = fields
        self.client = ollama.Client(host=ollama_host)
        self.cache_path = Path(cache_path) if cache_path else None
        self.summarizer_model = summarizer_model
        self.max_chars = max_chars

        # estatísticas da sessão
        self._stats = {"hits": 0, "misses": 0, "embed_time": 0.0, "summarize_time": 0.0}

        self._cache: dict[str, np.ndarray] = {}
        self._load_cache()

    # ── Cache ────────────────────────────────────────────────────────────────

    def _load_cache(self):
        if self.cache_path and self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=True)
            self._cache = {k: data[k] for k in data.files}
            log.info(f"Cache carregado: {len(self._cache)} embeddings de '{self.cache_path}'")
        else:
            log.info("Nenhum cache encontrado — todos os artigos serão embedados.")

    def _save_cache(self):
        if self.cache_path and self._cache:
            np.savez(self.cache_path, **self._cache)
            log.info(f"Cache salvo: {len(self._cache)} embeddings em '{self.cache_path}'")

    # ── Resumo automático ────────────────────────────────────────────────────

    def _summarize(self, text: str) -> str:
        log.info(f"  ↳ resumindo ({len(text)} chars) via '{self.summarizer_model}'...")
        t0 = time.perf_counter()
        prompt = (
            "Resuma o texto abaixo em até 1000 caracteres, mantendo os pontos mais relevantes e"
            "focando em fatos financeiros e impacto no mercado. "
            "Responda apenas com o resumo, sem introdução.\n\n"
            f"{text[:6000]}"
        )
        response = self.client.generate(model=self.summarizer_model, prompt=prompt)
        elapsed = time.perf_counter() - t0
        self._stats["summarize_time"] += elapsed
        log.info(f"  ↳ resumo gerado em {_fmt(elapsed)}")
        return response["response"].strip()

    # ── Embedding de um texto ────────────────────────────────────────────────

    def _embed_one(self, key: str, text: str) -> np.ndarray:
        if key in self._cache:
            self._stats["hits"] += 1
            return self._cache[key]

        self._stats["misses"] += 1

        if len(text) > self.max_chars:
            if self.summarizer_model:
                text = self._summarize(text)
            else:
                text = text[:self.max_chars]

        t0 = time.perf_counter()
        response = self.client.embed(model=self.model, input=text)
        elapsed = time.perf_counter() - t0
        self._stats["embed_time"] += elapsed

        vec = np.array(response["embeddings"][0], dtype=np.float32)
        self._cache[key] = vec
        return vec

    # ── Agregação diária ─────────────────────────────────────────────────────

    @staticmethod
    def _weighted_mean(
        embeddings: list[np.ndarray],
        timestamps: list[datetime],
    ) -> np.ndarray:
        order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
        weights = np.arange(1, len(order) + 1, dtype=np.float32)
        matrix = np.stack([embeddings[i] for i in order])
        weighted = (matrix * weights[:, None]).sum(axis=0)
        return weighted / weights.sum()

    # ── API pública ──────────────────────────────────────────────────────────

    def embed_articles(self, articles: list[dict]) -> dict[str, np.ndarray]:
        """
        Embeda uma lista de artigos e agrupa por dia (YYYY-MM-DD).
        Loga progresso, tempo por artigo e estimativa de conclusão.
        """
        total = len(articles)
        log.info(f"Iniciando embedding de {total} artigos com '{self.model}'")

        by_day: dict[str, list[tuple[datetime, np.ndarray]]] = {}

        t_total = time.perf_counter()
        times_per_article: list[float] = []

        for i, art in enumerate(articles, 1):
            text = _build_text(art, self.fields, self.max_chars)
            if not text:
                log.warning(f"  [{i}/{total}] artigo id={art.get('id')} sem texto — pulando")
                continue

            key = str(art.get("id", hash(text)))
            cached = key in self._cache

            t0 = time.perf_counter()
            vec = self._embed_one(key, text)
            elapsed = time.perf_counter() - t0
            times_per_article.append(elapsed)

            dt = _parse_date(art["date"])
            day = dt.strftime("%Y-%m-%d")
            by_day.setdefault(day, []).append((dt, vec))

            # log a cada artigo com ETA
            status = "cache" if cached else "embed"
            avg = sum(times_per_article) / len(times_per_article)
            eta = avg * (total - i)
            log.info(
                f"  [{i:>4}/{total}] {day} | {status} | {_fmt(elapsed):>6} | "
                f"ETA {_fmt(eta)} | chars={len(text)}"
            )

        elapsed_total = time.perf_counter() - t_total
        self._save_cache()
        self._log_summary(total, elapsed_total)

        # agregar por dia
        daily: dict[str, np.ndarray] = {}
        for day, items in by_day.items():
            timestamps = [t for t, _ in items]
            vecs       = [v for _, v in items]
            daily[day] = self._weighted_mean(vecs, timestamps)

        log.info(f"Embeddings prontos: {len(daily)} dias únicos")
        return daily

    def _log_summary(self, total: int, elapsed_total: float):
        hits   = self._stats["hits"]
        misses = self._stats["misses"]
        e_time = self._stats["embed_time"]
        s_time = self._stats["summarize_time"]

        log.info("─" * 55)
        log.info(f"  Total de artigos  : {total}")
        log.info(f"  Cache hits        : {hits}  ({hits/max(total,1)*100:.0f}%)")
        log.info(f"  Embeddings novos  : {misses}")
        log.info(f"  Tempo embed       : {_fmt(e_time)}")
        if s_time:
            log.info(f"  Tempo resumos     : {_fmt(s_time)}")
        log.info(f"  Tempo total       : {_fmt(elapsed_total)}")
        if misses:
            log.info(f"  Média por embed   : {_fmt(e_time / misses)}")
        log.info("─" * 55)

    def to_dataframe(self, articles: list[dict]) -> pd.DataFrame:
        daily = self.embed_articles(articles)
        if not daily:
            return pd.DataFrame()

        index = pd.to_datetime(sorted(daily.keys()))
        matrix = np.stack([daily[d.strftime("%Y-%m-%d")] for d in index])

        cols = [f"emb_{i}" for i in range(matrix.shape[1])]
        df = pd.DataFrame(matrix, index=index, columns=cols)
        df.index.name = "Date"
        return df

    def merge_with_prices(
        self,
        price_features: pd.DataFrame,
        articles: list[dict],
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        emb_df = self.to_dataframe(articles)
        merged = price_features.join(emb_df, how="left")

        if fill_method:
            emb_cols = [c for c in merged.columns if c.startswith("emb_")]
            merged[emb_cols] = (
                merged[emb_cols].ffill() if fill_method == "ffill"
                else merged[emb_cols].bfill()
            )

        return merged.dropna()