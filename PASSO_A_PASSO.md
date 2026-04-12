# Passo a Passo do Pipeline de Predição de Preços com NLP

## Visao Geral

Este projeto e um TCC (Trabalho de Conclusao de Curso) que investiga se **noticias financeiras brasileiras** podem melhorar a previsao de direcao de precos de acoes da B3. O pipeline coleta noticias do InfoMoney, gera embeddings via modelos de linguagem, combina com dados de mercado do Yahoo Finance, e treina modelos de classificacao binaria (sobe/desce em 21 dias).

**Acao principal analisada:** ITUB4 (Itau Unibanco)

---

## Etapa 1 — Coleta de Noticias (`1.news/`)

### Arquivo: `extractor.py`
### Notebook: `extrator_de_noticias.ipynb`

### O que faz
Coleta artigos do portal InfoMoney usando a REST API do WordPress (`/wp-json/wp/v2/posts`).

### Passo a passo

1. **Configuracao**: Define as acoes alvo (`PETR4`, `ITUB4`, `VALE3`) e o periodo de busca. Originalmente 5 anos (`meses_atras=60`), expandido em US-010 para o maximo disponivel com modo incremental (`incremental=True`) que faz append sem duplicatas usando o campo `id` como chave.

2. **Extracao paralela**: A funcao `extrair_varias_acoes()` dispara uma thread por acao usando `ThreadPoolExecutor(max_workers=3)`.

3. **Paginacao**: Para cada acao, a classe `ExtratorDeNoticias` percorre todas as paginas da API (100 artigos por pagina), usando o parametro `modified_after` para filtrar por data.

4. **Retentativas**: Cada requisicao HTTP tem ate 3 tentativas com backoff linear (1s, 2s). Status codes `429`, `500-504` disparam retry automatico.

5. **Pre-processamento dos artigos**:
   - `_strip_html()`: remove tags HTML e decodifica entidades (`&amp;` → `&`)
   - `_normalize_text()`: normaliza whitespace e trunca textos longos
   - `_extract_keywords()`: extrai keywords do schema Yoast SEO (JSON-LD)

6. **Estruturacao**: Cada artigo e convertido em um dataclass `Artigo` com campos: `id`, `date`, `title`, `link`, `excerpt`, `content`, `author_name`, `hat`, `categories`, `tags`, `keywords`.

7. **Persistencia**: Salva em JSON (`{ticker}_noticias.json`) e CSV (`{ticker}_noticias.csv`).

### Volumes coletados

| Ativo | Artigos |
|-------|---------|
| PETR4 | ~1.775 |
| VALE3 | ~1.525 |
| ITUB4 | ~798 → expandido para 2.572 (5 anos) |

---

## Etapa 2 — Engenharia de Features (`2.stocks/`)

### Arquivos: `yahoo_finance.py`, `news_embedder.py`
### Notebook: `index.ipynb`

### Passo a passo

#### 2.1 — Features de mercado (`yahoo_finance.py`)

1. **Busca de dados**: A classe `MarketData` usa `yfinance` para baixar dados OHLCV de `ITUB4.SA`. O periodo padrao foi alterado de `period='5y'` para `period='max'` (US-010) para buscar todo o historico disponivel.

2. **Calculo de features tecnicas** via `MarketData.features()`:
   - `Close`, `Volume` — preco de fechamento e volume negociado
   - `return` — retorno diario percentual (`pct_change()`)
   - `ma7`, `ma21` — medias moveis de 7 e 21 dias
   - `std21` — desvio padrao de 21 dias (proxy de volatilidade)
   - `lag_1` a `lag_5` — valores defasados do fechamento

   Total: **11 features de preco** por dia.

#### 2.2 — Embeddings de noticias (`news_embedder.py`)

1. **Modelo de embedding**: `qwen3-embedding:4b` (rodando via Ollama local, host `http://100.68.136.71:11434`). Gera vetores de **1.024 dimensoes** por artigo.

2. **Construcao do texto**: Para cada artigo, concatena `title | excerpt | content` (truncado em 50.000 caracteres).

3. **Resumo automatico** (opcional): Para textos muito longos, um modelo summarizer (`lfm2.5-thinking`) gera um resumo de ate 1.000 caracteres antes do embedding.

4. **Cache de embeddings**: Embeddings calculados sao salvos em `embeddings_cache.npz` (NumPy compressed). Na reexecucao, artigos ja embedados sao reutilizados sem nova chamada ao modelo.

5. **Agregacao diaria**: Quando ha multiplas noticias no mesmo dia, os embeddings sao agregados via **media ponderada por recencia** (`_weighted_mean`): noticias mais recentes no dia recebem peso maior (pesos lineares 1, 2, 3...).

   - 2.572 artigos → 1.115 dias unicos com noticias

#### 2.3 — Merge temporal (`merge_with_prices()`)

1. **Left join por data**: Combina features de preco (DataFrame indexado por data de mercado) com embeddings diarios.

2. **Forward-fill**: Para dias de mercado sem noticias publicadas, propaga o ultimo embedding disponivel (`ffill`).

3. **Drop de NaN**: Remove linhas com valores ausentes.

### Dataset final

- **Arquivo**: `dataset_full.csv`
- **Dimensoes**: 1.227 dias x 1.035 features (11 preco + 1.024 embeddings)
- **Periodo**: 2021-04-28 a 2026-03-26 (~5 anos)

---

## Etapa 3 — Treinamento de Modelos (`3.model_traning/`)

### Arquivos: `lstm_classifier.py`, `xgboost_baseline.py`, `transformer_classifier.py`
### Notebook: `index.ipynb`

### Passo a passo

#### 3.1 — Preparacao do dataset (`build_dataset()`)

1. **Target binario**: `1` se `Close[t+21] > Close[t]` (preco sobe em 21 dias uteis), `0` caso contrario.
   - Distribuicao: 59% sobe / 41% desce

2. **PCA nos embeddings**: Reduz 1.024 dimensoes de embedding para **32 componentes** via PCA.
   - Variancia explicada: ~61,4%

3. **Normalizacao**: `StandardScaler` fitado **apenas nos primeiros 70% dos dados** (treino) para evitar data leakage.

4. **Janelas temporais**: Cria sequencias de **30 dias** para entrada dos modelos sequenciais (BiLSTM, Transformer).
   - Shape resultante: `(1.176, 30, 43)` — 1.176 amostras, janela de 30 dias, 43 features (11 preco + 32 PCA)

#### 3.2 — Validacao walk-forward

Split cronologico (sem shuffle) para respeitar a ordem temporal:

| Conjunto | Proporcao | Periodo |
|----------|-----------|---------|
| Treino | 70% (823 amostras) | 2021-06 a 2024-09 |
| Validacao | 15% (176 amostras) | 2024-09 a 2025-06 |
| Teste | 15% (177 amostras) | 2025-06 a 2026-02 |

#### 3.3 — Modelo 1: BiLSTM Original (`lstm_classifier.py`)

**Arquitetura:**
```
Input (30, 43) → BiLSTM(2 camadas, 128 hidden, 30% dropout)
→ Dropout(0.3) → Dense(64) → ReLU → Dense(1) → Sigmoid
```

**Estrategia de treino:**
- Otimizador: Adam (lr=0,001, weight_decay=1e-4)
- Loss: BCELoss com balanceamento de classes via `pos_weight`
- Early stopping: patience=10 epocas
- LR scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Gradient clipping: max_norm=1.0

**Resultado:** ROC-AUC = **0.4432**, Acuracia = 38.4% (early stop na epoch 11)

#### 3.4 — Modelo 2: BiLSTM Reduzido

**Arquitetura:** BiLSTM(1 camada, 64 hidden, 20% dropout) — versao simplificada para testar se menos complexidade melhora generalizacao.

**Resultado:** ROC-AUC = **0.5051**, Acuracia = 30.5% (early stop na epoch 14)

#### 3.5 — Modelo 3: XGBoost Baseline (`xgboost_baseline.py`)

**Objetivo:** Baseline classico sem dependencia temporal para comparar com modelos sequenciais.

**Configuracao:**
- `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`
- `early_stopping_rounds=20`
- Sem janelamento temporal (dataset flat/tabular)

**Resultado:** ROC-AUC = **0.6103**, Acuracia = 30.8% (melhor iteracao: 9)

#### 3.6 — Modelo 4: Transformer (`transformer_classifier.py`)

**Arquitetura:**
```
Input (30, 43) → Linear(43→64) → Positional Encoding (sinusoidal)
→ TransformerEncoder(2 camadas, 4 cabecas, d_model=64, d_ff=256)
→ Mean Pooling → Dense(32) → ReLU → Dropout(0.3) → Dense(1) → Sigmoid
```

**Resultado:** ROC-AUC = **0.5684**, Acuracia = 69.5% (early stop na epoch 11)

---

## Tabela Comparativa de Resultados

| Modelo | ROC-AUC | Acuracia | F1 (macro) |
|--------|---------|----------|------------|
| BiLSTM Original (2L/128h) | 0.4432 | 38.4% | 0.3777 |
| BiLSTM Reduzido (1L/64h) | 0.5051 | 30.5% | 0.2338 |
| XGBoost Baseline | **0.6103** | 30.8% | 0.2353 |
| Transformer (2L/d64) | 0.5684 | **69.5%** | 0.4100 |

---

## Diagnostico e Conclusoes

1. **Todos os modelos ficam proximo ao acaso** (ROC-AUC ~0.5), indicando que o sinal preditivo e fraco.

2. **O problema nao e arquitetural**: tanto modelos sequenciais (BiLSTM, Transformer) quanto tabulares (XGBoost) apresentam desempenho similar.

3. **O dataset expandido ajudou**: com 1.227 dias (vs. 230 iniciais), os modelos sairam de overfitting extremo (ROC-AUC 0.10) para resultados proximos ao aleatorio (~0.5-0.6).

4. **A hipotese da eficiencia de mercado** pode ser o fator limitante: precos de acoes liquidas como ITUB4 sao dificeis de prever com base apenas em noticias e indicadores tecnicos.

---

## US-010 — Expansao do Historico de Dados

Para mitigar o underfitting e aumentar o volume de amostras, as seguintes alteracoes foram implementadas:

1. **`yahoo_finance.py`**: periodo padrao alterado de `period='5y'` para `period='max'`
2. **`extractor.py`**: adicionado metodo `carregar_existentes()` para append incremental sem duplicatas (dedup por `id`)
3. **`extrair_varias_acoes()`**: novo parametro `incremental=True` que carrega JSON existente antes de salvar
4. **Meta**: expandir de 1.227 para >3.600 dias de dados
5. **Cache**: `embeddings_cache.npz` e reaproveitado automaticamente pelo `NewsEmbedder`

### Como executar a expansao

```python
# 1. Notícias (incremental)
from extractor import extrair_varias_acoes
extratores = extrair_varias_acoes(
    acoes=["PETR4", "ITUB4", "VALE3"],
    meses_atras=12 * 17,  # desde ~2009
    incremental=True,
)

# 2. Preços (period='max' agora é o default)
from yahoo_finance import MarketData
md = MarketData("ITUB4.SA")
X_price = md.features(lags=5)  # usa period='max'

# 3. Embeddings (cache reaproveitado)
from news_embedder import NewsEmbedder
ne = NewsEmbedder(model="qwen3-embedding:4b", cache_path="embeddings_cache.npz")
X_full = ne.merge_with_prices(X_price, articles)
```

---

## Possiveis Extensoes

- Incorporar dados de sentimento mais granulares (ex: FinBERT-PT-BR ao inves de embeddings brutos)
- Testar horizontes de previsao diferentes (5, 10, 42 dias)
- Adicionar features macroeconomicas (Selic, cambio, indices globais)
- Experimentar ensemble dos modelos
- Usar dados de noticias de outros tickers (PETR4, VALE3) ja coletados

---

## Stack Tecnologica

| Componente | Tecnologia |
|------------|------------|
| Linguagem | Python 3.13 |
| Coleta de noticias | `requests` + WordPress REST API |
| Dados de mercado | `yfinance` |
| Embeddings | Ollama (`qwen3-embedding:4b`) |
| Modelos deep learning | PyTorch (BiLSTM, Transformer) |
| Modelo classico | XGBoost |
| Pre-processamento | scikit-learn (PCA, StandardScaler) |
| Manipulacao de dados | pandas, numpy |
| Visualizacao | matplotlib |
| Execucao | Jupyter Notebooks |
