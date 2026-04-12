# Diagrama do Pipeline de Predição de Preços

```mermaid
flowchart TD
    A["🌐 API InfoMoney<br/>/wp-json/wp/v2/posts<br/><i>Notícias de PETR4, VALE3, ITUB4</i>"] --> B

    B["📰 extractor.py<br/><b>ExtratorDeNoticias</b><br/>Coleta + limpeza HTML<br/>+ extração de keywords"] --> C["📄 {ticker}_noticias.json<br/>{ticker}_noticias.csv"]

    C --> E

    D["📈 yahoo_finance.py<br/><b>MarketData.features(period='max')</b><br/>OHLCV + retorno + MA7/21<br/>+ volatilidade + lags 1-5"] --> F

    E["🧠 news_embedder.py<br/><b>NewsEmbedder</b><br/>Embeddings via qwen3-embedding:4b<br/>Agregação diária (weighted mean)<br/>Cache em embeddings_cache.npz"] --> F

    F["🔗 merge_with_prices()<br/>Left join temporal<br/>+ forward-fill para dias sem notícias"] --> G

    G["📊 dataset_full.csv<br/>1.227 dias × 1.035 features<br/>(11 price + 1.024 emb)<br/>US-010: meta >3.600 dias com period=max"]

    G --> H["🏗️ build_dataset()<br/>Target binário (sobe em 21 dias)<br/>PCA 1024→32 dims<br/>Normalização + janelas de 30 dias"]

    H --> I["🤖 Modelos"]

    I --> J["BiLSTM<br/>lstm_classifier.py<br/>2 camadas, 128 hidden<br/>ROC-AUC: 0.4432"]
    I --> K["BiLSTM Reduzido<br/>1 camada, 64 hidden<br/>ROC-AUC: 0.5051"]
    I --> L["XGBoost<br/>xgboost_baseline.py<br/>Baseline clássico<br/>ROC-AUC: 0.6103"]
    I --> M["Transformer<br/>transformer_classifier.py<br/>2 camadas, d_model=64<br/>ROC-AUC: 0.5684"]

    J --> N["📋 Métricas<br/>ROC-AUC, Precision, Recall<br/>F1, Confusion Matrix<br/>Curvas de loss e ROC"]
    K --> N
    L --> N
    M --> N
```

## Fluxo de dados resumido

1. **Coleta** → API InfoMoney fornece artigos em JSON via REST API do WordPress
2. **Extração** → `extractor.py` limpa HTML, normaliza texto, extrai keywords
3. **Features de mercado** → `yahoo_finance.py` busca OHLCV e calcula indicadores técnicos
4. **Embeddings** → `news_embedder.py` gera vetores de 1024 dims via Ollama, agrega por dia
5. **Merge** → Combina preços + embeddings por data, forward-fill para dias sem notícias
6. **Preparação** → PCA, normalização, janelamento temporal
7. **Modelos** → BiLSTM (original e reduzido), XGBoost, Transformer
8. **Avaliação** → Métricas de classificação com validação walk-forward
