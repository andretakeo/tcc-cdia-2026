# Predição de Direção de Preços de Ações com NLP e Sentimento de Notícias

**Trabalho de Conclusão de Curso**
**Ação analisada:** ITUB4 (Itaú Unibanco)

---

## Resumo

Este projeto investiga se notícias financeiras brasileiras podem melhorar a previsão de direção de preços de ações da B3. O pipeline completo coleta notícias do portal InfoMoney, extrai representações semânticas via modelos de linguagem, combina com dados de mercado do Yahoo Finance, e treina modelos de classificação binária (sobe/desce).

O trabalho passou por 5 etapas iterativas, onde cada resultado informou a direção seguinte. O principal achado foi que **features de sentimento financeiro específico** (via FinBERT-PT-BR) superam embeddings genéricos de alta dimensionalidade, com o modelo Transformer alcançando ROC-AUC de 0.709 — significativamente acima do acaso.

---

## Etapa 1 — Coleta de Notícias (`1.news/`)

A primeira etapa consiste na construção de um corpus de notícias financeiras brasileiras. O módulo `extractor.py` acessa a REST API do WordPress do portal InfoMoney (`/wp-json/wp/v2/posts`), que é uma das principais fontes de notícias financeiras no Brasil.

Para cada ticker (PETR4, ITUB4, VALE3), o sistema realiza buscas paginadas com até 100 artigos por página, percorrendo todo o histórico disponível. Cada requisição HTTP possui mecanismo de retentativa: até 3 tentativas com backoff linear (1s, 2s), tratando erros de timeout, conexão e status codes 429/5xx.

Os artigos passam por um pipeline de pré-processamento: remoção de tags HTML, decodificação de entidades HTML, normalização de whitespace e extração de keywords do schema Yoast SEO (JSON-LD). Cada artigo é estruturado com campos de título, data, link, resumo, conteúdo completo, autor, editoria, categorias, tags e keywords.

O sistema suporta extração incremental: antes de buscar novas notícias, carrega artigos já existentes do JSON e faz deduplicação por ID, evitando reprocessamento. A extração de múltiplos tickers ocorre em paralelo via `ThreadPoolExecutor`.

**Volumes coletados:** PETR4: ~1.775 artigos | VALE3: ~1.525 artigos | ITUB4: ~2.572 artigos

---

## Etapa 2 — Engenharia de Features (`2.stocks/`)

Nesta etapa, os dados textuais são transformados em representações numéricas e combinados com features de mercado.

### Features de mercado

O módulo `yahoo_finance.py` utiliza a biblioteca `yfinance` para obter dados OHLCV (abertura, máxima, mínima, fechamento, volume) do ticker ITUB4.SA. A partir desses dados brutos, são calculadas 11 features técnicas por dia: preço de fechamento, volume, retorno diário percentual, médias móveis de 7 e 21 dias, desvio padrão de 21 dias (proxy de volatilidade) e 5 valores defasados do fechamento (lags 1 a 5).

### Embeddings de notícias

O módulo `news_embedder.py` transforma cada artigo em um vetor numérico de 1.024 dimensões usando o modelo `qwen3-embedding:4b` rodando via Ollama. Para cada artigo, o texto de entrada é a concatenação de título, resumo e conteúdo (truncado em 50.000 caracteres).

Quando há múltiplas notícias no mesmo dia, os embeddings são agregados via média ponderada por recência: notícias mais recentes no dia recebem peso maior. Os embeddings calculados são cacheados em arquivo NumPy comprimido (`embeddings_cache.npz`), eliminando recálculos em reexecuções.

### Merge temporal

As features de preço e os embeddings diários são combinados via left join por data. Para dias de mercado sem notícias publicadas, aplica-se forward-fill nos embeddings, propagando o último embedding disponível.

**Dataset resultante:** 1.227 dias × 1.035 features (11 de preço + 1.024 de embeddings), período de 2021-04 a 2026-03.

---

## Etapa 3 — Treinamento de Modelos (`3.model_traning/`)

Com o dataset construído, esta etapa treina e compara 4 modelos de classificação binária para prever se o preço sobe ou desce em 21 dias úteis.

### Preparação dos dados

O target binário é definido como 1 se `Close[t+21] > Close[t]`, resultando em um desbalanceamento de 59% sobe / 41% desce. Os 1.024 embeddings são reduzidos para 32 componentes via PCA (61,4% de variância explicada). A normalização é feita com StandardScaler fitado apenas nos primeiros 70% dos dados (treino) para evitar data leakage. Para os modelos sequenciais, são criadas janelas de 30 dias, resultando em 1.176 sequências de shape (30, 43).

### Validação walk-forward

O split é cronológico, respeitando a ordem temporal dos dados: 70% treino (2021-06 a 2024-09), 15% validação (2024-09 a 2025-06), 15% teste (2025-06 a 2026-02). Não é feito shuffle para evitar vazamento de informação futura.

### Modelos avaliados

**BiLSTM Original** — LSTM bidirecional com 2 camadas, 128 hidden units e 30% de dropout. Treinado com Adam, early stopping e gradient clipping. Resultado: ROC-AUC = 0.443.

**BiLSTM Reduzido** — Versão simplificada com 1 camada e 64 hidden units para testar se menos complexidade melhora a generalização. Resultado: ROC-AUC = 0.505.

**XGBoost Baseline** — Modelo clássico tabular sem dependência temporal. Usa o mesmo dataset com PCA e normalização, mas sem janelamento. Resultado: ROC-AUC = 0.610.

**Transformer** — Encoder com 2 camadas de atenção multi-cabeça (4 heads), projeção linear para d_model=64, positional encoding sinusoidal e mean pooling temporal. Resultado: ROC-AUC = 0.568.

### Conclusão da Etapa 3

Todos os modelos apresentaram desempenho próximo ou abaixo do acaso (ROC-AUC ~0.5), indicando que a combinação de features de preço com embeddings genéricos de alta dimensionalidade não é suficiente para prever a direção de preços. O XGBoost obteve o melhor resultado (AUC 0.610), mas ainda insuficiente para uso prático. A conclusão inicial foi de que o problema poderia ser intrinsecamente difícil (hipótese de eficiência de mercado).

---

## Etapa 4 — Análise de Sentimento com FinBERT-PT-BR (`4.finbert-br/`)

A hipótese desta etapa é que **embeddings genéricos** (1.024 dimensões) contêm muito ruído semântico, e que uma representação mais focada — sentimento financeiro — poderia ser mais informativa.

### Extração de sentimento

O modelo FinBERT-PT-BR, treinado especificamente em textos financeiros brasileiros, classifica cada artigo como POSITIVO, NEGATIVO ou NEUTRO. A inferência é feita em batch (32 artigos por vez) usando título + resumo como entrada (mais focado que o conteúdo completo, e dentro do limite de 512 tokens do BERT).

Além da classe discreta, são extraídos os logits brutos do modelo (3 valores: positivo, negativo, neutro), que preservam mais informação que a classificação final. Os artigos são processados para todos os tickers: ITUB4 (2.572 artigos), PETR4 (1.775) e VALE3 (1.525).

### Agregação diária

Os logits são agregados por dia: média dos logits positivo/negativo/neutro, média da classe de sentimento e contagem de artigos. Isso produz 5 features de sentimento por dia, que são combinadas com as 11 features de preço via left join temporal com forward-fill.

### Treinamento com sentimento

Os mesmos 4 modelos da Etapa 3 foram retreinados, agora com 16 features (11 preço + 5 sentimento) ao invés de 1.035 (11 preço + 1.024 embeddings). Não há necessidade de PCA pois a dimensionalidade já é baixa.

### Resultados comparativos

| Modelo | AUC (Etapa 3, Embeddings) | AUC (Etapa 4, Sentimento) | Δ |
|--------|:-------------------------:|:-------------------------:|:-:|
| BiLSTM Original | 0.443 | 0.500 | +0.057 |
| BiLSTM Reduzido | 0.505 | 0.477 | -0.028 |
| XGBoost | 0.610 | 0.670 | +0.060 |
| Transformer | 0.568 | **0.709** | **+0.141** |

O Transformer foi o grande vencedor: ROC-AUC de 0.709, accuracy de 76.3% e F1(Sobe) de 0.85. Quando prevê queda, acerta 100% das vezes (precision perfeita, porém conservador com recall de 20%).

### Conclusão da Etapa 4

O resultado mais importante do projeto: **5 features de sentimento financeiro específico superam 1.024 embeddings genéricos**. O FinBERT-PT-BR funciona como um filtro de sinal, comprimindo o texto em dimensões diretamente relevantes para o mercado financeiro. O gargalo identificado na Etapa 3 não era arquitetural — era a representação textual.

---

## Etapa 5 — Ajuste de Threshold e Experimentos (`5.threshold-tuning/`)

Os modelos da Etapa 4, exceto o Transformer, previam 100% de uma única classe com threshold padrão de 0.5. Esta etapa investigou se o ajuste de threshold e variações no dataset poderiam resolver o problema.

### Experimento 5.1: Ajuste de threshold

Para cada modelo, o threshold foi otimizado no conjunto de validação maximizando o F1-score, com varredura de 0.10 a 0.90. Resultado: 3 dos 4 modelos tiveram threshold ótimo no limite inferior (0.10), indicando que apenas preveem a classe majoritária. O XGBoost manteve AUC consistente (0.670) mas sem discriminação real com nenhum threshold.

### Experimento 5.2: Horizonte de 5 dias + Feature Engineering

Duas mudanças foram testadas simultaneamente:

**Horizonte reduzido (21 → 5 dias):** A hipótese era que o sentimento tem impacto de curto prazo. O target ficou mais equilibrado (54/46 vs 59/41), mas o sinal preditivo enfraqueceu — todos os modelos pioraram.

**Features temporais derivadas (16 novas features):** Médias móveis de sentimento (7 e 21 dias), variação de sentimento (delta vs MA7), razão positivo/negativo, volume acumulado de notícias e volatilidade do sentimento. Total de 32 features.

| Modelo | AUC (h=21, 16 feat) | AUC (h=5, 32 feat) | Δ |
|--------|:-------------------:|:------------------:|:-:|
| BiLSTM Original | 0.500 | 0.433 | -0.067 |
| BiLSTM Reduzido | 0.477 | 0.460 | -0.017 |
| Transformer | 0.709 | 0.588 | -0.121 |
| XGBoost | 0.670 | 0.528 | -0.142 |

Resultado negativo, porém informativo: a análise de feature importance do XGBoost revelou que **6 das 10 features mais importantes são derivadas de sentimento**, com a média de 21 dias do logit negativo como a #1. Isso confirma que as features derivadas capturam sinal relevante, mas o horizonte de 5 dias é curto demais para o tipo de notícia do InfoMoney (que reflete tendências de médio prazo).

### Conclusão da Etapa 5

O ajuste de threshold não substitui poder discriminativo. O horizonte de 5 dias piorou todos os modelos, confirmando que o sinal de sentimento do InfoMoney se manifesta no médio prazo (~21 dias). O melhor resultado do projeto permanece sendo o Transformer com horizonte de 21 dias e 5 features base de sentimento FinBERT (AUC 0.709).

---

## Resultado Final

| Configuração | Melhor Modelo | ROC-AUC | Accuracy |
|-------------|:-------------:|:-------:|:--------:|
| Embeddings Ollama (1.024 dims) | XGBoost | 0.610 | 30.8% |
| Sentimento FinBERT (5 dims) | **Transformer** | **0.709** | **76.3%** |
| Sentimento + horizon 5 dias | Transformer | 0.588 | 40.1% |

O pipeline mais eficaz combina: notícias do InfoMoney → sentimento via FinBERT-PT-BR → 5 features diárias → Transformer com horizonte de 21 dias.

---

## Principais Aprendizados

1. **A representação textual importa mais que a arquitetura do modelo.** Na Etapa 3, todos os modelos falharam com embeddings genéricos. Na Etapa 4, o mesmo Transformer alcançou AUC 0.709 apenas trocando a representação.

2. **Menos dimensões, mais sinal.** 5 features de sentimento financeiro superaram 1.024 embeddings genéricos. A redução de dimensionalidade feita por um modelo especializado (FinBERT-PT-BR) funciona como filtro de ruído.

3. **O horizonte temporal importa.** Sentimento de notícias do InfoMoney reflete tendências de médio prazo. Horizonte de 5 dias é curto demais; 21 dias captura melhor o impacto.

4. **Threshold padrão (0.5) pode ser enganoso.** Modelos com calibração ruim podem parecer inúteis com threshold fixo mas ter capacidade de ranking (como o XGBoost com AUC 0.670 mas accuracy de 30.8%).

5. **Accuracy é uma métrica traiçoeira** em datasets desbalanceados. O BiLSTM Original com accuracy de 69.5% estava apenas prevendo a classe majoritária. ROC-AUC é mais confiável para avaliar poder discriminativo.

---

## Stack Tecnológica

| Componente | Tecnologia |
|------------|------------|
| Linguagem | Python 3.13 |
| Coleta de notícias | `requests` + WordPress REST API |
| Dados de mercado | `yfinance` |
| Embeddings genéricos | Ollama (`qwen3-embedding:4b`, 1.024 dims) |
| Sentimento financeiro | FinBERT-PT-BR (HuggingFace Transformers) |
| Modelos deep learning | PyTorch (BiLSTM, Transformer) |
| Modelo clássico | XGBoost |
| Pré-processamento | scikit-learn (PCA, StandardScaler) |
| Manipulação de dados | pandas, NumPy |
| Visualização | matplotlib |
| Execução | Jupyter Notebooks |

---

## Estrutura do Repositório

```
tcc-takeo/
├── 1.news/                     # Coleta de notícias do InfoMoney
│   ├── extractor.py            # Módulo de extração
│   └── extrator_de_noticias.ipynb
├── 2.stocks/                   # Features de mercado e embeddings
│   ├── yahoo_finance.py        # Features técnicas de preço
│   ├── news_embedder.py        # Embeddings via Ollama
│   └── index.ipynb
├── 3.model_traning/            # Treinamento com embeddings Ollama
│   ├── lstm_classifier.py      # BiLSTM
│   ├── xgboost_baseline.py     # XGBoost
│   ├── transformer_classifier.py
│   └── index.ipynb
├── 4.finbert-br/               # Sentimento FinBERT-PT-BR
│   ├── index.ipynb             # Extração de sentimento
│   ├── model_training.ipynb    # Retreino dos 4 modelos
│   └── ANALISE_RESULTADOS.md
├── 5.threshold-tuning/         # Ajuste de threshold e experimentos
│   ├── index.ipynb             # Threshold tuning
│   ├── horizon5_features.ipynb # Horizonte 5 dias + feature eng.
│   ├── ANALISE_RESULTADOS.md
│   └── ANALISE_HORIZON5.md
├── PASSO_A_PASSO.md            # Documentação técnica detalhada
├── pipeline.md                 # Diagrama do pipeline (Mermaid)
└── README.md                   # Este arquivo
```
