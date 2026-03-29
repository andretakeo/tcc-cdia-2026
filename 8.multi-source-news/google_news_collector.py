"""
Coletor de notícias do Google News para ITUB4 / Itaú Unibanco.

Utiliza duas abordagens para robustez:
1. Biblioteca gnews (principal)
2. RSS do Google News (fallback)

As notícias coletadas são salvas em JSON para uso no pipeline de predição de preços.
"""

import argparse
import json
import logging
import time
from datetime import datetime

import feedparser
from gnews import GNews
from urllib.parse import quote

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Queries padrão para buscar notícias sobre Itaú/ITUB4
DEFAULT_QUERIES = ["ITUB4", "Itaú Unibanco", "Itaú ações"]


def collect_gnews(query: str, period: str = "1y", max_results: int = 200) -> list[dict]:
    """
    Coleta notícias usando a biblioteca gnews.

    Args:
        query: Termo de busca (ex: "ITUB4")
        period: Período de busca (ex: "1y", "6m", "7d")
        max_results: Número máximo de resultados

    Returns:
        Lista de dicionários com as notícias coletadas
    """
    logger.info(f"Coletando notícias via gnews: query='{query}', period={period}, max={max_results}")

    gn = GNews(language="pt", country="BR", period=period, max_results=max_results)
    raw_articles = gn.get_news(query)

    if not raw_articles:
        logger.warning(f"Nenhuma notícia encontrada via gnews para '{query}'")
        return []

    articles = []
    for art in raw_articles:
        # Parsear a data publicada
        pub_date = art.get("published date", "")
        try:
            dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = pub_date

        publisher = art.get("publisher", {})
        if isinstance(publisher, dict):
            publisher_name = publisher.get("title", "Desconhecido")
        else:
            publisher_name = str(publisher)

        articles.append({
            "date": date_str,
            "title": art.get("title", ""),
            "source": publisher_name,
            "publisher": publisher_name,
            "link": art.get("url", ""),
            "description": art.get("description", ""),
            "text": None,  # Preenchido depois, se solicitado
        })

    logger.info(f"gnews retornou {len(articles)} notícias para '{query}'")
    return articles


def collect_rss(query: str) -> list[dict]:
    """
    Coleta notícias via RSS do Google News (fallback).

    Args:
        query: Termo de busca

    Returns:
        Lista de dicionários com as notícias coletadas
    """
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    logger.info(f"Coletando notícias via RSS: query='{query}'")

    feed = feedparser.parse(url)

    if not feed.entries:
        logger.warning(f"Nenhuma notícia encontrada via RSS para '{query}'")
        return []

    articles = []
    for entry in feed.entries:
        # Parsear data do RSS
        pub_date = entry.get("published", "")
        try:
            dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = pub_date

        # Fonte vem no campo source do RSS
        source = entry.get("source", {})
        if isinstance(source, dict):
            source_name = source.get("title", "Desconhecido")
        else:
            source_name = str(source) if source else "Desconhecido"

        articles.append({
            "date": date_str,
            "title": entry.get("title", ""),
            "source": source_name,
            "publisher": source_name,
            "link": entry.get("link", ""),
            "description": entry.get("summary", ""),
            "text": None,
        })

    logger.info(f"RSS retornou {len(articles)} notícias para '{query}'")
    return articles


def fetch_full_text(url: str) -> str | None:
    """
    Tenta extrair o texto completo de um artigo usando newspaper3k.

    Lida graciosamente com falhas (paywalls, bloqueios, timeouts).

    Args:
        url: URL do artigo

    Returns:
        Texto completo ou None em caso de falha
    """
    try:
        from newspaper import Article

        article = Article(url, language="pt")
        article.download()
        article.parse()

        if article.text and len(article.text) > 50:
            return article.text
        return None
    except Exception as e:
        logger.debug(f"Falha ao extrair texto de {url}: {e}")
        return None


def collect_all(
    queries: list[str] | None = None,
    period: str = "1y",
    max_results: int = 200,
    fetch_text: bool = True,
    limit: int | None = None,
) -> list[dict]:
    """
    Coleta notícias de múltiplas queries, deduplica por URL e opcionalmente
    busca o texto completo dos artigos.

    Args:
        queries: Lista de termos de busca (usa DEFAULT_QUERIES se None)
        period: Período de busca para gnews
        max_results: Máximo de resultados por query no gnews
        fetch_text: Se deve tentar extrair texto completo
        limit: Limitar número total de artigos (para testes)

    Returns:
        Lista de artigos deduplicados
    """
    if queries is None:
        queries = DEFAULT_QUERIES

    # Coletar de todas as fontes
    all_articles = []
    for query in queries:
        # Abordagem 1: gnews
        try:
            articles = collect_gnews(query, period=period, max_results=max_results)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Erro no gnews para '{query}': {e}")

        # Abordagem 2: RSS (fallback / complementar)
        try:
            articles = collect_rss(query)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Erro no RSS para '{query}': {e}")

    # Deduplicar por URL
    seen_urls = set()
    unique_articles = []
    for art in all_articles:
        url = art.get("link", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(art)

    logger.info(f"Total após deduplicação: {len(unique_articles)} artigos (de {len(all_articles)} coletados)")

    # Limitar se necessário (modo teste)
    if limit is not None:
        unique_articles = unique_articles[:limit]
        logger.info(f"Limitado a {limit} artigos (modo teste)")

    # Buscar texto completo dos artigos
    if fetch_text:
        logger.info("Iniciando extração de texto completo...")
        for i, art in enumerate(unique_articles):
            url = art.get("link", "")
            if url:
                logger.info(f"  [{i + 1}/{len(unique_articles)}] Extraindo texto: {art['title'][:60]}...")
                art["text"] = fetch_full_text(url)

                # Delay de 1-2 segundos entre requisições para evitar bloqueios
                if i < len(unique_articles) - 1:
                    time.sleep(1.5)

        textos_ok = sum(1 for a in unique_articles if a["text"])
        logger.info(f"Texto completo extraído de {textos_ok}/{len(unique_articles)} artigos")

    # Ordenar por data (mais recente primeiro)
    unique_articles.sort(key=lambda x: x.get("date", ""), reverse=True)

    return unique_articles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coletor de notícias do Google News para ITUB4")
    parser.add_argument("--test", action="store_true", help="Modo teste: coleta apenas 10 artigos")
    parser.add_argument("--no-text", action="store_true", help="Não extrair texto completo")
    parser.add_argument("--output", default="itub4_google_news.json", help="Arquivo de saída (default: itub4_google_news.json)")
    args = parser.parse_args()

    # Configurações baseadas nos argumentos
    limit = 10 if args.test else None
    fetch_text = not args.no_text

    logger.info("=" * 60)
    logger.info("Coletor de Notícias Google News - ITUB4 / Itaú Unibanco")
    logger.info("=" * 60)

    if args.test:
        logger.info(">>> MODO TESTE: limitado a 10 artigos <<<")

    # Coletar notícias
    articles = collect_all(
        fetch_text=fetch_text,
        limit=limit,
        max_results=20 if args.test else 200,
    )

    # Salvar em JSON
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"Salvo {len(articles)} artigos em {output_path}")
    logger.info("Concluído!")
