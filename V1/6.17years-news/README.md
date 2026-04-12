# Etapa 6 — Histórico Estendido (17 anos)

## Objetivo

Testar se um histórico significativamente maior (≈17 anos vs ≈5 anos da Etapa 4) melhora a capacidade preditiva. A hipótese é que mais regimes de mercado (crise 2008, ciclo de commodities, COVID, ciclo Selic recente) tornem o modelo mais robusto.

## Pipeline

1. **Coleta estendida** — `extractor.py` (Etapa 1) + extensão temporal para cobrir desde 2009.
2. **Preços** — `yahoo_finance.py` (versão local nesta pasta) com `period='max'`.
3. **Sentimento** — mesmo procedimento da Etapa 4 (FinBERT-PT-BR, 5 features diárias).
4. **Treinamento** — mesmos 4 modelos (BiLSTM original, BiLSTM reduzido, XGBoost, Transformer), mesmo split walk-forward.

## Variáveis

- Tickers: ITUB4 (principal), VALE3 (auxiliar)
- Dataset principal: `itub4_daily_sentiment_17y.csv`
- Sentimento por artigo: `itub4_noticias.json`, `vale3_noticias.json`

## Discussão de regimes

Períodos atravessados pelo dataset estendido:
- **2009–2013** — pós-crise financeira global, recuperação
- **2014–2016** — recessão brasileira, Lava Jato, impeachment
- **2017–2019** — recuperação lenta, reformas
- **2020–2021** — COVID e resposta monetária
- **2022–2026** — ciclo de Selic, inflação, eleições

Cada regime impõe relação distinta entre sentimento de notícias e movimento de preços. Isso é simultaneamente uma oportunidade (mais variedade de exemplos) e um risco (não-estacionariedade).

## Notas

A execução completa requer reprocessar embeddings/sentimento para o histórico inteiro (custoso). Os artefatos principais (`*.png`, `*_17y.csv`) registram os resultados.

## Arquivos

- `index.ipynb` — pipeline completo (coleta → sentimento → treino → avaliação)
- `yahoo_finance.py` — versão local com `period='max'`
- `itub4_daily_sentiment_17y.csv` — features diárias estendidas
- `lstm_results.png`, `transformer_17y.png`, `xgboost_roc_17y.png`, `roc_17years.png` — gráficos
