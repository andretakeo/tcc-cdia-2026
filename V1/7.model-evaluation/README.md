# Etapa 7 — Avaliação Cruzada Consolidada

## Objetivo

Comparar de forma consistente todos os modelos das Etapas 3–6, em um único lugar, com utilitários compartilhados. Isso permite responder com clareza:

- Qual configuração (representação × arquitetura × horizonte × histórico) é a melhor?
- Quanto de cada melhoria vem da representação textual vs do volume de dados vs da arquitetura?
- As diferenças observadas são estatisticamente significativas?

## Estrutura

```
7.model-evaluation/
├── shared/                       # Utilitários compartilhados (split, métricas, plots)
├── results/                      # Saídas (CSVs, PNGs, relatórios)
├── stage3_ollama_embeddings.ipynb  # Reavaliação Etapa 3 (embeddings Ollama 1024d)
├── stage4_finbert_4y.ipynb         # Reavaliação Etapa 4 (FinBERT, 4–5 anos)
├── stage5_horizon5.ipynb           # Reavaliação Etapa 5 (horizonte 5 dias)
├── stage6_finbert_17y.ipynb        # Reavaliação Etapa 6 (histórico 17 anos)
└── BALANCEAMENTO_DE_CLASSES.md     # Discussão de desbalanceamento e métricas
```

## Padrão de avaliação

Todos os notebooks seguem o mesmo protocolo:
1. Carregar o dataset apropriado da etapa.
2. Usar split walk-forward 70/15/15 sem leakage.
3. Treinar os 4 modelos com hiperparâmetros documentados na etapa original.
4. Reportar **ROC-AUC** (métrica primária), **F1** por classe, accuracy e matriz de confusão.
5. Salvar resultados em `results/` para consolidação.

## Tabela consolidada (resultados principais)

| Configuração | Melhor modelo | ROC-AUC | Accuracy |
|---|:---:|:---:|:---:|
| Stage 3 — Ollama embeddings (1024d) | XGBoost | 0,610 | 30,8% |
| Stage 4 — FinBERT sentimento (5 feat) | **Transformer** | **0,709** | **76,3%** |
| Stage 5 — Horizonte 5d (32 feat) | Transformer | 0,588 | 40,1% |
| Stage 6 — FinBERT 17 anos | a consolidar | — | — |

## Balanceamento de classes

Ver `BALANCEAMENTO_DE_CLASSES.md`. Resumo:
- Target binário desbalanceado: 59% sobe / 41% desce (h=21).
- Accuracy é métrica enganosa neste cenário — um modelo que sempre prevê "sobe" obtém 59% de accuracy sem capacidade preditiva real.
- ROC-AUC é a métrica primária. F1 por classe complementa identificando colapso para classe majoritária.
- BCEWithLogitsLoss com `pos_weight` é usado nos modelos PyTorch para mitigar o desbalanceamento.

## Próximos passos sugeridos

- **Bootstrap CIs** sobre as métricas de teste (1000 reamostragens) para comparar configurações com significância estatística.
- **Replicação por ticker** (PETR4, VALE3) para confirmar generalização.
- **Backtest** simples do Transformer-Etapa4 com custos de transação para traduzir AUC em retorno financeiro.
