# Etapa 8 — Coleta Multi-Fonte de Notícias

## Motivação

Todos os experimentos anteriores usam **InfoMoney como única fonte**. Isso impõe dois riscos:

1. **Viés de fonte** — o vocabulário, o tom editorial e a cobertura temática do InfoMoney podem enviesar o sinal de sentimento.
2. **Lacunas de cobertura** — eventos importantes podem ter sido cobertos por outras fontes (Valor, Estadão, Reuters, agências regulatórias) e estar ausentes do dataset.

Esta etapa expande para **múltiplas fontes** e re-aplica o pipeline de sentimento da Etapa 4.

## Fontes adicionadas

### Google News (`google_news_collector.py`)

- Busca notícias por ticker/empresa via Google News.
- Decodifica URLs do Google News para a fonte original.
- Tenta extrair o **texto completo** do artigo (fallback: título).
- Resolve corretamente o nome da fonte real (não "Google News (...)").

### CVM (`cvm_collector.py`)

- Coleta de fatos relevantes e comunicados oficiais publicados no portal da Comissão de Valores Mobiliários.
- Fonte primária regulatória — maior peso de sinal por evento.

## Pipeline

```
Coletores (InfoMoney + Google News + CVM)
    ↓ artigos brutos
Limpeza (HTML strip, dedup por URL/título)
    ↓
FinBERT-PT-BR (mesma inferência da Etapa 4)
    ↓ logits + classe por artigo
Agregação diária por fonte e total
    ↓
multi_source_daily_sentiment.csv
```

## Saídas

- `all_articles_with_sentiment.json` — todos os artigos coletados com sentimento por artigo.
- `multi_source_daily_sentiment.csv` — features agregadas por dia (com colunas por fonte).
- `results/` — gráficos e relatórios de avaliação.

## Notebooks

- `coleta_completa.ipynb` — pipeline completo de coleta + sentimento.
- `test_pipeline.ipynb` — testes parciais e validação dos coletores.

## Limitações conhecidas

- **Rate limits** do Google News e dos sites originais.
- **Paywall** em algumas fontes (Valor, FSP) — extração de full text pode falhar e cair no fallback de título.
- **Qualidade variável** do texto extraído via parsers genéricos (boilerplate, scripts inline).
- **Deduplicação cross-fonte** é heurística (mesma notícia republicada por agências).

## Perguntas a investigar

1. Adicionar Google News + CVM ao InfoMoney **aumenta o AUC** do Transformer da Etapa 4, ou apenas adiciona ruído?
2. Há fontes com **sinal individualmente mais forte** que o InfoMoney?
3. Comunicados da CVM (eventos discretos, alta confiabilidade) têm efeito desproporcional?
