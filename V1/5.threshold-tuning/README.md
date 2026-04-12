# Etapa 5 — Threshold Tuning e Experimento de Horizonte

Esta etapa investiga duas hipóteses para melhorar os modelos da Etapa 4:

1. O threshold padrão (0,5) está mascarando capacidade de discriminação real?
2. Um horizonte mais curto (5 dias) captura melhor o impacto do sentimento?

## Experimento 5.1 — Ajuste de threshold

**Procedimento:** para cada modelo, varredura de threshold de 0,10 a 0,90, otimizando F1 no conjunto de validação.

**Resultado:** 3 dos 4 modelos têm threshold ótimo no limite inferior (0,10), revelando que apenas preveem a classe majoritária. O XGBoost mantém AUC consistente (0,670) mas sem discriminação real em nenhum threshold.

**Lição metodológica:** *threshold tuning não substitui poder discriminativo*. Um modelo com AUC ≈ 0,5 não vira preditor útil ao mover o threshold — só troca um tipo de erro por outro.

## Experimento 5.2 — Horizonte 5 dias + 16 features derivadas

Duas mudanças simultâneas:

**Horizonte reduzido (21 → 5 dias)** — alvo mais balanceado (54/46 vs 59/41), mas o sinal preditivo enfraquece em todos os modelos.

**Features temporais derivadas (16 novas, total 32):**
- Médias móveis 7d e 21d dos logits de sentimento
- Delta vs MA7 (variação curta)
- Razão positivo/negativo
- Volume acumulado de notícias
- Volatilidade do sentimento (rolling std)

| Modelo | AUC (h=21, 16 feat) | AUC (h=5, 32 feat) | Δ |
|---|:---:|:---:|:---:|
| BiLSTM Original | 0.500 | 0.433 | -0.067 |
| BiLSTM Reduzido | 0.477 | 0.460 | -0.017 |
| Transformer | 0.709 | 0.588 | -0.121 |
| XGBoost | 0.670 | 0.528 | -0.142 |

## Resultado positivo dentro do experimento negativo

A análise de **feature importance do XGBoost** revela que **6 das 10 features mais importantes são derivadas de sentimento**. A média de 21 dias do logit negativo é a feature #1.

Isso confirma que as features derivadas capturam sinal relevante — só não no horizonte de 5 dias.

## Conclusão

- O ajuste de threshold não substitui poder discriminativo.
- O sentimento das notícias do InfoMoney se manifesta no **médio prazo (~21 dias)**, não em 5 dias.
- O melhor resultado do projeto permanece o da Etapa 4: **Transformer + sentimento FinBERT + horizonte 21d → AUC 0,709**.

## Arquivos

- `index.ipynb` — threshold tuning (Experimento 5.1)
- `horizon5_features.ipynb` — horizonte 5 dias + feature engineering (Experimento 5.2)
- `ANALISE_RESULTADOS.md` — análise detalhada do Experimento 5.1
- `ANALISE_HORIZON5.md` — análise detalhada do Experimento 5.2
- `threshold_search*.png` — varreduras de threshold
- `confusion_matrices_optimized.png` — confusões pós-otimização
- `feature_importance_h5.png` — importância das features no XGBoost h=5
- `roc_*.png` — curvas ROC
