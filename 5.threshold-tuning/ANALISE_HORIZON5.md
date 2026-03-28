# Análise dos Resultados — Horizonte 5 dias + Feature Engineering

**Hipóteses testadas:**
1. Reduzir o horizonte de 21 → 5 dias captura melhor o impacto de curto prazo do sentimento
2. Features derivadas (médias móveis, deltas, razões) enriquecem o sinal de sentimento

**Dataset:** 1.207 dias × 32 features (11 preço + 5 sentimento base + 16 sentimento derivadas)
**Target:** sobe/desce em 5 dias úteis (balance: 54% sobe / 46% desce — mais equilibrado que os 59/41 do horizonte 21)

---

## Observações Positivas

### 1. Balance do target melhorou significativamente

| Horizonte | % Sobe | % Desce | Desbalanceamento |
|:---------:|:------:|:-------:|:----------------:|
| 21 dias | 59% | 41% | 18pp |
| 5 dias | 54% | 46% | 8pp |

Com horizonte de 5 dias, o target é quase balanceado. Isso elimina uma das armadilhas anteriores: a accuracy por classe majoritária agora é ~54% (vs ~69% antes), então prever tudo "Sobe" já não infla as métricas.

### 2. BiLSTM Reduzido mostrou sinais de discriminação real

No relatório de classificação, o BiLSTM Reduzido obteve:
- **Precision(Desce) = 0.42, Recall(Desce) = 0.23** — pela primeira vez um LSTM previu *alguma* queda
- Macro avg F1 = 0.49 — próximo de um classificador informado

Isso não apareceu nos experimentos anteriores com horizonte 21.

### 3. XGBoost parou na iteração 0

O early stopping parou imediatamente (`best_iteration: 0`), indicando que **nenhuma divisão melhorou a validação**. Com 32 features, o modelo não encontrou splits informativos — o sinal é disperso demais.

### 4. Features de sentimento dominam o XGBoost

Top 10 features por importância:

| # | Feature | Tipo | Importance |
|---|---------|------|:----------:|
| 1 | `mean_logit_neg_ma21` | Sentimento | 0.0532 |
| 2 | `std21` | Preço | 0.0476 |
| 3 | `lag_4` | Preço | 0.0456 |
| 4 | `mean_logit_neu_ma7` | Sentimento | 0.0449 |
| 5 | `ma7` | Preço | 0.0447 |
| 6 | `lag_3` | Preço | 0.0445 |
| 7 | `mean_sentiment_ma7` | Sentimento | 0.0426 |
| 8 | `n_articles_sum21` | Sentimento | 0.0425 |
| 9 | `mean_logit_neg_ma7` | Sentimento | 0.0397 |
| 10 | `mean_logit_neu_ma21` | Sentimento | 0.0380 |

**6 das 10 features mais importantes são de sentimento**, com destaque para `mean_logit_neg_ma21` (média de 21 dias do logit negativo) como a #1. Isso confirma que as features derivadas capturam sinal — o problema é que esse sinal é insuficiente para gerar previsões confiáveis.

---

## Resultados no Teste

| Modelo | ROC-AUC | Accuracy (t=0.5) | Discrimina? |
|--------|:-------:|:-----------------:|:-----------:|
| BiLSTM Original (2L/128h) | 0.433 | 59.9% | Não |
| BiLSTM Reduzido (1L/64h) | 0.460 | 56.5% | Parcial |
| Transformer | 0.588 | 40.1% | Não |
| XGBoost | 0.528 | 55.8% | Parcial |

**Todos os thresholds ótimos caíram em 0.10** — nenhum modelo discrimina de verdade com threshold otimizado (todos preveem classe única).

---

## Comparação com Etapa 4 (Horizonte 21 dias, 16 features)

| Modelo | AUC (h=21) | AUC (h=5) | Δ |
|--------|:----------:|:---------:|:-:|
| BiLSTM Original | 0.500 | 0.433 | -0.067 |
| BiLSTM Reduzido | 0.477 | 0.460 | -0.017 |
| Transformer | **0.709** | 0.588 | **-0.121** |
| XGBoost | **0.670** | 0.528 | **-0.142** |

**O horizonte de 5 dias piorou todos os modelos.** A queda mais expressiva foi no XGBoost (-0.142) e no Transformer (-0.121) — justamente os que tinham os melhores resultados com horizonte 21.

---

## Diagnóstico

### Por que o horizonte curto piorou?

1. **Ruído domina no curto prazo.** Em 5 dias, o preço é mais influenciado por microestrutura de mercado, fluxo de ordens e eventos pontuais do que por sentimento de notícias. O target com horizonte 21 suaviza esse ruído.

2. **O balance melhorou, mas o sinal piorou.** O target mais equilibrado (54/46) é melhor para treino, mas a relação sentimento→preço em 5 dias é mais fraca que em 21 dias. As notícias financeiras do InfoMoney refletem tendências de médio prazo, não de curtíssimo prazo.

3. **Feature engineering não compensou.** As 16 features derivadas diluíram o sinal ao invés de concentrá-lo — 32 features para ~1.200 amostras aumenta o risco de overfitting sem adicionar informação proporcional.

### Por que o XGBoost parou na iteração 0?

Com 32 features e sinal fraco, o regularization do XGBoost (`max_depth=4`, `subsample=0.8`) impediu qualquer split de melhorar o logloss na validação. O modelo retornou a prior da classe, que é basicamente prever ~54% de probabilidade para tudo.

---

## Conclusões

1. **Horizonte de 5 dias é curto demais** para o tipo de notícia do InfoMoney. O sinal de sentimento se manifesta em prazos mais longos (~21 dias), onde o Transformer alcançou AUC 0.709.

2. **Feature engineering de sentimento adicionou complexidade sem benefício.** As features derivadas são relevantes individualmente (aparecem no top 10 do XGBoost), mas coletivamente não melhoram a predição — indicativo de que o sinal está saturado com as 5 features base.

3. **O melhor resultado continua sendo Transformer + horizonte 21 dias + 5 features base** (AUC 0.709) da Etapa 4.

4. **Próximos passos mais promissores:**
   - Testar horizonte intermediário (10 dias)
   - Combinar sentimento FinBERT + embeddings Ollama (representações complementares)
   - Multi-ticker para triplicar os dados de treino
   - Seed fixo + múltiplas execuções para estabilizar os resultados do Transformer
