# Etapa 4 — Sentimento Financeiro com FinBERT-PT-BR

## Hipótese

Embeddings genéricos de 1.024 dimensões (Etapa 3) contêm muito ruído semântico. Uma representação compacta e específica do domínio financeiro — sentimento via FinBERT-PT-BR — deve ser mais informativa para prever direção de preços.

## Modelo

**FinBERT-PT-BR** — modelo BERT treinado em textos financeiros em português brasileiro. Classifica texto em três classes: `POSITIVO`, `NEGATIVO`, `NEUTRO`, e fornece logits brutos (3 valores por artigo).

## Pipeline

1. **Carregamento** — `transformers.AutoModelForSequenceClassification` a partir do diretório local `FinBERT-PT-BR/`.
2. **Inferência em batch** — 32 artigos por vez, input = `título + resumo` (truncado em 512 tokens).
3. **Saída por artigo** — classe predita + logits brutos `[pos, neg, neu]`.
4. **Agregação diária** — média dos logits, média da classe e contagem de artigos → **5 features por dia**.
5. **Merge temporal** — left join com features de preço (11 features OHLCV/técnicas), forward-fill em dias sem notícias.

Tickers processados: ITUB4 (2.572 artigos), PETR4 (1.775), VALE3 (1.525).

## Treinamento

Os 4 modelos da Etapa 3 (BiLSTM original, BiLSTM reduzido, XGBoost, Transformer) foram retreinados com **16 features** (11 preço + 5 sentimento) ao invés de 1.035. Mesmo split walk-forward (70/15/15), sem PCA.

## Resultados

| Modelo | AUC (Etapa 3, embeddings) | AUC (Etapa 4, sentimento) | Δ |
|---|:---:|:---:|:---:|
| BiLSTM Original | 0.443 | 0.500 | +0.057 |
| BiLSTM Reduzido | 0.505 | 0.477 | -0.028 |
| XGBoost | 0.610 | 0.670 | +0.060 |
| **Transformer** | 0.568 | **0.709** | **+0.141** |

Melhor configuração: **Transformer** com `AUC=0.709`, `accuracy=76,3%`, `F1(Sobe)=0,85`, `precision(Desce)=1,00` (recall(Desce)=0,20).

## Conclusão central

> **5 features de sentimento financeiro específico superam 1.024 embeddings genéricos.**

O FinBERT-PT-BR atua como filtro de sinal, comprimindo o texto em dimensões diretamente relevantes para o mercado. O gargalo da Etapa 3 não era arquitetural — era a representação textual.

## Arquivos

- `index.ipynb` — extração de sentimento (carregamento do modelo, inferência, agregação)
- `model_training.ipynb` — retreinamento dos 4 modelos com features de sentimento
- `ANALISE_RESULTADOS.md` — análise detalhada modelo a modelo
- `*_daily_sentiment.csv` — features diárias por ticker
- `*_noticias_sentiment.json` — sentimento por artigo
- `*.png` — gráficos de loss, ROC e matrizes de confusão
