"""
Coletor de Fatos Relevantes da CVM para Itaú Unibanco (ITUB4).

Este módulo baixa os dados abertos da CVM (Comissão de Valores Mobiliários),
filtra os Fatos Relevantes do Itaú Unibanco e extrai o texto completo
dos documentos para uso em análise de sentimento e predição de preços.

Fonte: https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/
"""

import argparse
import io
import json
import logging
import time
import zipfile
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# URL base dos arquivos IPE da CVM
BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{year}.zip"

# Configurações de retry
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # segundos: 2, 4, 8


def _request_with_retry(url: str, timeout: int = 30) -> requests.Response:
    """Faz requisição HTTP com lógica de retry e backoff exponencial."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                logger.error(f"Falha após {MAX_RETRIES} tentativas para {url}: {e}")
                raise
            wait = BACKOFF_FACTOR ** attempt
            logger.warning(f"Tentativa {attempt} falhou para {url}: {e}. Aguardando {wait}s...")
            time.sleep(wait)


def download_year(year: int) -> pd.DataFrame:
    """
    Baixa e descompacta o arquivo ZIP de um ano específico da CVM.

    Cada ZIP contém CSVs com separador ';' e encoding 'latin-1'.
    Retorna um DataFrame com todos os registros do ano.
    """
    url = BASE_URL.format(year=year)
    logger.info(f"Baixando dados de {year}: {url}")

    resp = _request_with_retry(url, timeout=60)

    # Descompacta o ZIP em memória e lê todos os CSVs
    frames = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                logger.info(f"  Lendo arquivo: {name}")
                with zf.open(name) as f:
                    df = pd.read_csv(
                        f,
                        sep=";",
                        encoding="latin-1",
                        dtype=str,  # lê tudo como string para evitar problemas
                        on_bad_lines="skip",
                    )
                    frames.append(df)

    if not frames:
        logger.warning(f"Nenhum CSV encontrado no ZIP de {year}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"  {len(combined)} registros carregados para {year}")
    return combined


def filter_itau_fatos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra o DataFrame para manter apenas Fatos Relevantes do Itaú Unibanco.

    - Nome_Companhia contém "ITAU UNIBANCO" (case insensitive, match parcial)
    - Categoria contém "Fato Relevante"
    """
    if df.empty:
        return df

    # Filtro pela empresa: Itaú Unibanco (nome pode ter acento: "ITAÚ UNIBANCO")
    mask_cia = df["Nome_Companhia"].str.contains("ITAU|ITAÚ", case=False, na=False, regex=True)
    mask_cia = mask_cia & df["Nome_Companhia"].str.contains("UNIBANCO", case=False, na=False)

    # Filtro pelo tipo de documento: Fato Relevante
    mask_doc = df["Categoria"].str.contains("Fato Relevante", case=False, na=False)

    filtered = df[mask_cia & mask_doc].copy()
    logger.info(f"  {len(filtered)} Fatos Relevantes do Itaú encontrados")
    return filtered


def fetch_document_text(link: str) -> str:
    """
    Busca o texto completo de um documento a partir do link da CVM.

    Os documentos geralmente são páginas HTML no sistema da CVM.
    Extrai apenas o texto visível, removendo tags HTML.
    """
    try:
        resp = _request_with_retry(link, timeout=30)
        # Tenta detectar o encoding correto
        resp.encoding = resp.apparent_encoding or "latin-1"
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts e estilos
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text
    except Exception as e:
        logger.warning(f"Erro ao buscar texto de {link}: {e}")
        return ""


def collect_all(start_year: int = 2009, end_year: int = 2026) -> pd.DataFrame:
    """
    Coleta todos os Fatos Relevantes do Itaú de um intervalo de anos.

    Baixa os dados de cada ano, filtra e concatena em um único DataFrame.
    Anos que falharem são ignorados (ex: ano futuro sem dados).
    """
    all_frames = []

    for year in range(start_year, end_year + 1):
        try:
            df = download_year(year)
            filtered = filter_itau_fatos(df)
            if not filtered.empty:
                all_frames.append(filtered)
        except Exception as e:
            logger.error(f"Erro ao processar ano {year}: {e}")
            continue

    if not all_frames:
        logger.warning("Nenhum Fato Relevante encontrado em todo o período")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(f"Total: {len(combined)} Fatos Relevantes coletados")
    return combined


def dataframe_to_json_records(df: pd.DataFrame, fetch_text: bool = True) -> list[dict]:
    """
    Converte o DataFrame filtrado para o formato JSON de saída.

    Para cada documento, opcionalmente busca o texto completo via LINK_DOC.
    """
    records = []

    for i, row in df.iterrows():
        # Usa Data_Referencia como data principal, com fallback para Data_Entrega
        date_str = row.get("Data_Referencia", "") or row.get("Data_Entrega", "")

        # Normaliza a data para formato ISO
        try:
            date_obj = pd.to_datetime(date_str)
            date_iso = date_obj.strftime("%Y-%m-%d")
        except Exception:
            date_iso = date_str

        link = row.get("Link_Download", "")

        # Busca o texto do documento se solicitado
        text = ""
        if fetch_text and link:
            logger.info(f"  Buscando texto do documento {i + 1}/{len(df)}: {link}")
            text = fetch_document_text(link)
            # Pausa entre requisições para não sobrecarregar o servidor
            time.sleep(0.5)

        record = {
            "date": date_iso,
            "title": "Fato Relevante",
            "category": str(row.get("Categoria", "Fato Relevante")),
            "source": "CVM",
            "link": link,
            "text": text,
        }
        records.append(record)

    return records


def main():
    """Ponto de entrada principal do coletor."""
    parser = argparse.ArgumentParser(
        description="Coletor de Fatos Relevantes da CVM para Itaú Unibanco (ITUB4)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Modo teste: baixa apenas dados de 2025 para validação rápida",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Não busca o texto completo dos documentos (mais rápido)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2009,
        help="Ano inicial da coleta (padrão: 2009)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="Ano final da coleta (padrão: 2026)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="itub4_fatos_relevantes.json",
        help="Arquivo de saída JSON (padrão: itub4_fatos_relevantes.json)",
    )

    args = parser.parse_args()

    if args.test:
        logger.info("=== MODO TESTE: apenas ano 2025 ===")
        df = collect_all(start_year=2025, end_year=2025)
    else:
        df = collect_all(start_year=args.start_year, end_year=args.end_year)

    if df.empty:
        logger.warning("Nenhum dado coletado. Encerrando.")
        return

    logger.info("Convertendo para formato JSON...")
    records = dataframe_to_json_records(df, fetch_text=not args.no_text)

    # Salva o arquivo JSON
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Salvo {len(records)} registros em {output_path}")


if __name__ == "__main__":
    main()
