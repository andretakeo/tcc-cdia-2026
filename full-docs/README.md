# Full Project Documentation

**Predição de Direção de Preços de Ações Brasileiras com Sentimento de Notícias Financeiras: Pipeline, Ilusão e Autocorreção Metodológica**

Author: André Takeo Loschner Fujiwara
Advisor: Prof. Eric Bacconi Gonçalves
Institution: PUC-SP — Ciência de Dados e Inteligência Artificial
Year: 2026

---

## Document Index

| Document | Description |
|---|---|
| [01-project-overview.md](01-project-overview.md) | Research question, thesis arc, and final conclusion |
| [02-data-collection.md](02-data-collection.md) | Stage 1: News collection from InfoMoney |
| [03-feature-engineering.md](03-feature-engineering.md) | Stage 2: Market data + Ollama embeddings |
| [04-initial-models.md](04-initial-models.md) | Stage 3: First models with generic embeddings |
| [05-finbert-sentiment.md](05-finbert-sentiment.md) | Stage 4: FinBERT-PT-BR sentiment extraction |
| [06-threshold-tuning.md](06-threshold-tuning.md) | Stage 5: Horizon and threshold experiments |
| [07-extended-history.md](07-extended-history.md) | Stage 6: 17-year historical data |
| [08-model-diagnostics.md](08-model-diagnostics.md) | Stage 7: SHAP, calibration, temporal stability |
| [09-multi-source-news.md](09-multi-source-news.md) | Stage 8: CVM + Google News exploration |
| [10-rigorous-validation.md](10-rigorous-validation.md) | Stage 9: The methodological investigation (Chapter 5) |
| [11-figures-catalog.md](11-figures-catalog.md) | Complete catalog of all figures with descriptions |
| [12-architecture-and-code.md](12-architecture-and-code.md) | Code architecture, shared modules, model specs |
| [13-conclusions.md](13-conclusions.md) | Final conclusions, limitations, and future work |

## How to Read This Documentation

The project follows a deliberate narrative arc:

1. **Stages 1-4** build the pipeline and produce an initially promising result (AUC 0.709)
2. **Stages 5-7** extend and diagnose the initial results
3. **Stage 8** explores multi-source news (exploratory, not in thesis)
4. **Stage 9** systematically dismantles the initial result through 1,500+ model runs

The central contribution is not the pipeline itself, but the **methodological self-correction** documented in Stage 9 / Chapter 5.
