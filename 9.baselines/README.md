# Etapa 9 — Baselines e Avaliação Rigorosa

## Por que esta etapa existe

Toda a narrativa do TCC depende de uma comparação que **ainda não foi feita honestamente**:

> O Transformer da Etapa 4, com 5 features de sentimento FinBERT-PT-BR, atinge AUC = 0,709.
> Mas qual seria o AUC de um modelo trivial que **não usa sentimento nenhum**?

Sem essa comparação, AUC = 0,709 é um número solto. Esta etapa o ancora.

## Conteúdo

- **`eval_utils.py`** — utilitários reusáveis: `walk_forward_split`, `bootstrap_auc_ci`, `evaluate_model`, `make_binary_target`. Estes serão importados pelas Etapas 10 (multi-ticker) e 11 (regime split).
- **`dumb_baseline.ipynb`** — treina 4 modelos (Logistic Regression, XGBoost, BiLSTM small, Transformer small) usando **apenas 5 features autoregressivas** (`return`, `lag_1`, `lag_5`, `Volume`, `std21`) — nenhuma feature derivada de notícias.
- **`results_dumb_baseline.csv`** — saída do notebook (gerada após execução).

## Protocolo

- **Target**: `1` se `Close[t+21] > Close[t]`, idêntico à Etapa 4.
- **Split**: walk-forward 70/15/15 (mesmo da Etapa 4).
- **Métrica primária**: ROC-AUC com **intervalo de confiança bootstrap 95%** (1.000 reamostragens).
- **Reportar sempre como**: `0.71 [0.64, 0.78]` — ponto + CI.

## Como usar o resultado

Compare o melhor AUC do dumb baseline contra **0,709** (Transformer + FinBERT da Etapa 4):

| Cenário | AUC do baseline | Implicação para a tese |
|---|---|---|
| **A** | < 0.60 | O sentimento adiciona valor real (Δ ≥ 0.10). Esta é a melhor notícia possível — reporte o delta explicitamente. |
| **B** | 0.60–0.68 | Valor marginal. Verifique se os CIs se sobrepõem — pode não ser estatisticamente significativo. |
| **C** | ≥ 0.68 | A maior parte do sinal vem dos preços. **Pivotar a tese**: o achado central deixa de ser "previsão" e passa a ser "comparação de representações textuais" (Etapa 3 vs Etapa 4). |

## Próximos passos (definidos no plano de 2 meses)

1. **Etapa 10 — Multi-ticker**: replicar Etapa 4 e este baseline em PETR4 e VALE3, usando os utilitários compartilhados aqui.
2. **Recalcular AUCs da Etapa 4** com `bootstrap_auc_ci` para que as comparações sejam simétricas (CI vs CI, não ponto vs CI).
3. **Etapa 11 — Regime split**: dividir o teste por volatilidade do IBOV (median split), reavaliar o melhor modelo dentro de cada regime.

## Como rodar

```bash
cd 9.baselines
jupyter notebook dumb_baseline.ipynb
# Run all cells. O CSV de resultados será salvo em results_dumb_baseline.csv.
```

Dependências: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `torch`. Tudo já presente no ambiente do projeto.
