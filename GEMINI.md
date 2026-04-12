# GEMINI.md

Este documento fornece contexto instrutivo e diretrizes para o desenvolvimento e manutenção do projeto de TCC sobre Predição de Direção de Preços de Ações com NLP e Sentimento de Notícias.

## Visão Geral do Projeto

Este projeto investiga o impacto de notícias financeiras brasileiras na previsão da direção dos preços das ações da B3 (foco inicial: ITUB4). O pipeline abrange desde a coleta de notícias até o treinamento de modelos de Deep Learning e Machine Learning clássico.

### Arquitetura do Pipeline (V1)

1.  **Coleta (`1.news/`):** Extração de notícias do InfoMoney via WordPress REST API.
2.  **Features (`2.stocks/`):** Dados de mercado (Yahoo Finance) e Embeddings de notícias (Ollama/qwen3).
3.  **Modelagem Inicial (`3.model_traning/`):** BiLSTM, XGBoost e Transformer usando embeddings genéricos + PCA.
4.  **Sentimento (`4.finbert-br/`):** Refinamento usando FinBERT-PT-BR para extrair sentimentos específicos (melhoria significativa de AUC).
5.  **Otimização (`5.threshold-tuning/`):** Ajuste de limiares de decisão e experimentos com horizontes temporais (5 vs 21 dias).

### Evolução (V2)

Foco em **Rigor Científico**:

- Expanding-Window Cross-Validation (validação temporal rigorosa).
- Correção de Look-Ahead Bias (ajuste fino do timestamp de notícias vs. fechamento do mercado).
- Estabelecimento de baselines ingênuas (Majoritário, Probabilístico, Inércia).

## Stack Tecnológica

- **Linguagem:** Python 3.13
- **Dados de Mercado:** `yfinance`
- **NLP:**
  - Embeddings: Ollama (`qwen3-embedding:4b`)
  - Sentimento: HuggingFace Transformers (`FinBERT-PT-BR`)
- **Modelos:**
  - Deep Learning: PyTorch (BiLSTM, Transformer)
  - Clássico: XGBoost
- **Processamento/Estatística:** pandas, numpy, scikit-learn (PCA, StandardScaler), matplotlib

## Convenções de Desenvolvimento

### Estrutura de Código

- **Dataclasses:** Uso de `dataclasses` para representar entidades (ex: `Artigo` em `extractor.py`).
- **Logging:** Uso sistemático do módulo `logging` para rastreamento de execução (ver `lstm_classifier.py`).
- **Tipagem:** Uso de Type Hints em funções e classes.
- **Paralelismo:** `ThreadPoolExecutor` para tarefas de I/O (extração de notícias).

### Manipulação de Dados Temporais

- **Split Cronológico:** NUNCA use `shuffle` em dados de treino/teste para evitar data leakage.
- **Forward-Fill:** Aplicado a embeddings/sentimentos para dias sem notícias, respeitando a ordem temporal.
- **Normalização:** `StandardScaler` deve ser fitado APENAS no conjunto de treino.

## Comandos e Uso

### Dependências

Não há um `requirements.txt` único. As dependências principais são:

```bash
pip install torch transformers xgboost yfinance pandas numpy scikit-learn matplotlib requests
```

### Execução de Módulos (Exemplos)

- **Extração de Notícias:**
  ```python
  from extractor import extrair_varias_acoes
  extrair_varias_acoes(acoes=["ITUB4"], meses_atras=60)
  ```
- **Treinamento BiLSTM:**
  ```bash
  python V1/3.model_traning/lstm_classifier.py
  ```

## Arquivos Chave e Localização

- `V1/1.news/extractor.py`: Lógica de coleta de notícias.
- `V1/2.stocks/yahoo_finance.py`: Engenharia de features de preço.
- `V1/4.finbert-br/index.ipynb`: Pipeline de sentimento com FinBERT.
- `V2/1.sientific-rigor/`: Implementações de validação temporal e baselines rigorosas.
- `PASSO_A_PASSO.md`: Guia técnico detalhado do pipeline.
- `README.md`: Resumo executivo e resultados finais.

## Práticas de Teste e Validação

- **Validação Walk-Forward:** Essencial para séries temporais.
- **Métricas Primárias:** ROC-AUC e F1-Score (especialmente para a classe minoritária "Queda").
- **Baseline:** Sempre comparar resultados com o XGBoost baseline e modelos ingênuos (V2).

---

_Este arquivo é mantido automaticamente como referência para agentes IA._
