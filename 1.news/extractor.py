import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import unescape
from typing import Any, Optional

import requests

BASE_URL = "https://www.infomoney.com.br/wp-json/wp/v2/posts"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "Referer": "https://www.infomoney.com.br/busca/",
}


def _strip_html(html_text: str) -> str:
    text = re.sub(r"<[^>]+>", "", html_text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_text(text: str, max_length: Optional[int] = None) -> str:
    clean = text.strip()
    if max_length is not None and len(clean) > max_length:
        return clean[:max_length].rstrip() + "..."
    return clean


def _extract_keywords(raw: dict[str, Any]) -> list[str]:
    graph = raw.get("yoast_head_json", {}).get("schema", {}).get("@graph", [])
    for item in graph:
        if isinstance(item, dict) and isinstance(item.get("keywords"), list):
            return item["keywords"]
    return []


def _parse_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str)


@dataclass
class Artigo:
    id: int
    date: str
    modified: str
    title: str
    link: str
    excerpt: str
    content: str
    author_id: int
    author_name: str
    hat: str
    categories: list[int]
    tags: list[int]
    keywords: list[str]

    @property
    def data(self) -> datetime:
        return _parse_date(self.date)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "date": self.date,
            "modified": self.modified,
            "title": self.title,
            "link": self.link,
            "excerpt": self.excerpt,
            "content": self.content,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "hat": self.hat,
            "categories": self.categories,
            "tags": self.tags,
            "keywords": self.keywords,
        }

    @classmethod
    def from_raw(cls, raw: dict[str, Any], content_max_length: Optional[int] = None) -> "Artigo":
        return cls(
            id=raw.get("id", 0),
            date=raw.get("date", ""),
            modified=raw.get("modified", ""),
            title=_strip_html(raw.get("title", {}).get("rendered", "")),
            link=raw.get("link", ""),
            excerpt=_normalize_text(_strip_html(raw.get("excerpt", {}).get("rendered", ""))),
            content=_normalize_text(
                _strip_html(raw.get("content", {}).get("rendered", "")),
                max_length=content_max_length,
            ),
            author_id=raw.get("author", 0),
            author_name=raw.get("yoast_head_json", {}).get("twitter_misc", {}).get("Written by", ""),
            hat=raw.get("meta", {}).get("post_hat", ""),
            categories=raw.get("categories", []),
            tags=raw.get("tags", []),
            keywords=_extract_keywords(raw),
        )


class ExtratorDeNoticias:
    """Extrai noticias da InfoMoney relacionadas a uma acao da bolsa por periodo."""

    def __init__(
        self,
        acao: str,
        data_inicio: Optional[str] = None,
        data_fim: Optional[str] = None,
        meses_atras: int = 12,
        per_page: int = 100,
        timeout: int = 20,
        retries: int = 3,
        content_max_length: Optional[int] = None,
    ) -> None:
        self.acao = acao.upper()
        self.per_page = min(per_page, 100)
        self.timeout = timeout
        self.retries = retries
        self.content_max_length = content_max_length
        self._artigos: list[Artigo] = []

        if data_fim:
            self.data_fim = datetime.fromisoformat(data_fim)
        else:
            self.data_fim = datetime.now()

        if data_inicio:
            self.data_inicio = datetime.fromisoformat(data_inicio)
        else:
            self.data_inicio = self.data_fim - timedelta(days=30 * meses_atras)

    @property
    def artigos(self) -> list[Artigo]:
        return list(self._artigos)

    @property
    def total(self) -> int:
        return len(self._artigos)

    def _request(self, params: dict[str, Any]) -> requests.Response:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                return resp
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_error = exc
                if attempt < self.retries:
                    print(f"  Retry {attempt}/{self.retries}: {exc}")
                    time.sleep(attempt)

        raise RuntimeError(f"Falha apos {self.retries} tentativas: {last_error}")

    def extrair(self) -> list[Artigo]:
        """Busca todas as noticias da acao dentro do periodo, usando modified_after da API."""
        self._artigos.clear()
        page = 1

        modified_after = self.data_inicio.strftime("%Y-%m-%dT%H:%M:%S")

        inicio_str = self.data_inicio.strftime("%Y-%m-%d")
        fim_str = self.data_fim.strftime("%Y-%m-%d")
        print(f"  [{self.acao}] Extraindo de {inicio_str} ate {fim_str}")

        while True:
            params = {
                "_embed": "wp:featuredmedia",
                "per_page": self.per_page,
                "page": page,
                "search": self.acao,
                "orderby": "modified",
                "order": "desc",
                "status": "publish",
                "context": "view",
                "modified_after": modified_after,
            }

            resp = self._request(params)

            if resp.status_code == 400:
                break
            resp.raise_for_status()

            data = resp.json()
            if not data:
                break

            for raw in data:
                artigo = Artigo.from_raw(raw, content_max_length=self.content_max_length)

                if artigo.data > self.data_fim:
                    continue

                self._artigos.append(artigo)

            total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
            print(f"  [{self.acao}] Pagina {page}/{total_pages} — {len(data)} artigos (coletados: {self.total})")

            if page >= total_pages:
                break
            page += 1

        print(f"  [{self.acao}] Total: {self.total} artigos no periodo")
        return self.artigos

    def salvar_json(self, path: Optional[str] = None) -> str:
        path = path or f"{self.acao.lower()}_noticias.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in self._artigos], f, ensure_ascii=False, indent=2)
        print(f"  [{self.acao}] Salvo em {path}")
        return path

    def salvar_csv(self, path: Optional[str] = None) -> str:
        path = path or f"{self.acao.lower()}_noticias.csv"
        fields = list(Artigo.__dataclass_fields__.keys())

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for artigo in self._artigos:
                row = artigo.to_dict()
                row["categories"] = ",".join(map(str, row["categories"]))
                row["tags"] = ",".join(map(str, row["tags"]))
                row["keywords"] = ",".join(row["keywords"])
                writer.writerow(row)

        print(f"  [{self.acao}] Salvo em {path}")
        return path

    def resumo(self) -> None:
        if not self._artigos:
            print(f"  [{self.acao}] Nenhum artigo extraido.")
            return

        mais_recente = self._artigos[0]
        mais_antigo = self._artigos[-1]
        print(f"\n  [{self.acao}] {self.total} artigos extraidos")
        print(f"  Periodo: {mais_antigo.date[:10]} ate {mais_recente.date[:10]}")
        print(f"  Mais recente: {mais_recente.title}")
        print(f"  Mais antigo:  {mais_antigo.title}")


def extrair_varias_acoes(
    acoes: list[str],
    data_inicio: Optional[str] = None,
    data_fim: Optional[str] = None,
    meses_atras: int = 12,
    per_page: int = 100,
    timeout: int = 20,
    retries: int = 3,
    content_max_length: Optional[int] = None,
    max_workers: int = 4,
    formato: str = "json",
) -> dict[str, ExtratorDeNoticias]:
    """Extrai noticias de multiplas acoes em paralelo usando threads."""

    def _processar(acao: str) -> ExtratorDeNoticias:
        ext = ExtratorDeNoticias(
            acao=acao,
            data_inicio=data_inicio,
            data_fim=data_fim,
            meses_atras=meses_atras,
            per_page=per_page,
            timeout=timeout,
            retries=retries,
            content_max_length=content_max_length,
        )
        ext.extrair()
        if formato == "csv":
            ext.salvar_csv()
        else:
            ext.salvar_json()
        return ext

    resultados: dict[str, ExtratorDeNoticias] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_processar, acao): acao for acao in acoes}

        for future in as_completed(futures):
            acao = futures[future]
            try:
                resultados[acao] = future.result()
            except Exception as exc:
                print(f"  [{acao}] ERRO: {exc}")

    print(f"\nResumo geral:")
    for acao, ext in resultados.items():
        print(f"  {acao}: {ext.total} artigos")

    return resultados
