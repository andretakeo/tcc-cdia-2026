# Analise Aprofundada — Stage 6: FinBERT Sentiment (17 anos)

## Contexto

Este estagio testa a hipotese: **mais dados historicos melhoram a previsao?** Em vez de usar apenas ~4 anos de noticias (Stage 4), coletamos 17 anos de artigos do InfoMoney (2009-2026), totalizando 12.891 artigos e 3.807 dias com sentimento. As features sao as mesmas do Stage 4 (11 preco + 5 sentimento = 16 features), com horizonte de 21 dias.

**Dataset:** 4.125 sequencias de 30 dias, balance 65.5% sobe / 34.5% desce. O balance e mais desigual que nos outros stages porque o periodo 2009-2026 inclui uma longa tendencia de alta do mercado brasileiro. Todos os modelos usam class weights para compensar.

**Expectativa:** O Stage 6 original tinha AUC de 0.616 (Transformer), ja abaixo do Stage 4 original (0.709). A hipotese era que padroes antigos (2009-2015) poderiam confundir os modelos.

**Split temporal:**
- Treino: 2009-2021 (~2.887 sequencias)
- Validacao: 2021-2023 (~618 sequencias)
- Teste: 2023-2026 (~620 sequencias)

---

## 1. Resultado geral: o pior de todos os stages

| Modelo | AUC | Accuracy | F1 (Sobe) | F1 (Desce) | ECE | Brier |
|--------|-----|----------|-----------|------------|-----|-------|
| TCN | **0.594** | 34.7% | 0.005 | 0.514 | 0.556 | 0.528 |
| Transformer | 0.537 | 34.5% | 0.000 | 0.513 | 0.394 | 0.376 |
| BiLSTM Reduzido | 0.537 | 37.4% | 0.093 | 0.522 | 0.477 | 0.454 |
| Random Forest | 0.529 | 37.2% | 0.066 | 0.523 | 0.425 | 0.410 |
| BiLSTM Original | 0.526 | 34.5% | 0.000 | 0.513 | 0.373 | 0.363 |
| Logistic Regression | 0.522 | 34.8% | 0.000 | 0.516 | 0.343 | 0.347 |
| XGBoost | 0.511 | 34.8% | 0.000 | 0.516 | 0.345 | 0.344 |

**Todos os modelos estao muito proximos do aleatorio (AUC 0.50).** Nenhum conseguiu classificar de forma util — F1 sobe e zero ou proximo de zero para todos.

### Comparacao com Stage 6 original:

| Modelo | AUC original | AUC re-treino | Diferenca |
|--------|:------------:|:-------------:|:---------:|
| Transformer | 0.616 | 0.537 | -0.079 |
| BiLSTM Original | 0.531 | 0.526 | -0.005 |
| BiLSTM Reduzido | 0.522 | 0.537 | +0.015 |
| XGBoost | 0.511 | 0.511 | 0.000 |

Os resultados sao consistentes com o original — ligeiramente piores para o Transformer, estaveis para os outros. A queda e menor que no Stage 4 (onde o Transformer caiu de 0.709 para 0.474), sugerindo que o resultado do Stage 6 original era mais robusto.

---

## 2. Matrizes de confusao: colapso quase total

As matrizes mostram o cenario mais extremo de colapso:

- **BiLSTM Original, Transformer, XGBoost, Logistic Regression:** 214-217 "desce" corretos, 405-407 "sobe" classificados como "desce". Zero previsoes de "sobe". Colapso completo.
- **BiLSTM Reduzido:** Previu 20 amostras como "sobe" (acertou 91% — precision altissima, mas recall de apenas 5%).
- **Random Forest:** Previu 14 como "sobe" (acertou 93%).
- **TCN:** Previu 1 unica amostra como "sobe" (e acertou).

O balance mais desequilibrado (65.5/34.5) torna o colapso ainda mais provavel — o "custo" de sempre prever "desce" e menor porque a classe "desce" e grande (34.5% de acuracia "gratis").

---

## 3. Distribuicao de previsoes: modelos extremamente pessimistas

- **BiLSTM Original:** Probabilidades entre 0.25 e 0.45. Nenhuma previsao ultrapassa 0.5. O modelo e consistentemente pessimista.

- **Transformer:** Concentrado entre 0.25 e 0.50. Similar ao BiLSTM mas com dispersao ligeiramente maior.

- **TCN:** A distribuicao mais extrema — quase tudo entre 0.05 e 0.15. O TCN esta dizendo "praticamente certeza que desce" para quase todas as amostras. Apesar disso, tem o melhor AUC (0.594), indicando que a ordenacao relativa esta parcialmente correta mesmo com calibracao pessima.

- **XGBoost:** Concentrado entre 0.20 e 0.45. Mais espalhado que nos stages anteriores, mas ainda abaixo do threshold.

- **Random Forest:** A distribuicao mais saudavel — vai de 0.1 a 0.8, com alguma separacao entre classes. Porem, a maioria das previsoes esta abaixo de 0.5.

- **Logistic Regression:** Quase tudo entre 0.05 e 0.25. Extremamente pessimista.

---

## 4. Estabilidade temporal: caos completo

O grafico de estabilidade temporal e o mais volatil de todos os stages:

- **Oscilacao extrema:** Todos os modelos alternam entre AUC 0.0 e 1.0 a cada trimestre. Nao ha nenhuma tendencia estavel.
- **Pior que os outros stages:** A amplitude de oscilacao e maior que nos Stages 3, 4 e 5b.
- **Nenhum modelo e consistentemente bom ou ruim:** Em um trimestre o BiLSTM e o melhor, no seguinte e o pior.

**Por que isso acontece:** O periodo de teste (2023-2026) e muito diferente do treino (2009-2021). O mercado brasileiro passou por transformacoes profundas nesse intervalo — mudancas de governo, ciclos de juros, pandemia, guerra na Ucrania, boom de commodities. Padroes aprendidos em 2009-2015 simplesmente nao se aplicam em 2023-2026.

---

## 5. SHAP: features de preco dominam, sentimento e ruido

### XGBoost — SHAP

As features mais importantes sao todas de preco:
1. **Close** — preco de fechamento
2. **ma21** — media movel 21 dias
3. **lag_1** — preco do dia anterior
4. **lag_5** — preco de 5 dias atras
5. **std21** — volatilidade

As features de sentimento (mean_logit_neg, mean_logit_neu, mean_sentiment, n_articles, mean_logit_pos) aparecem na metade inferior do ranking com impacto pequeno.

### XGBoost — Permutation Importance

**Close** e a unica feature com queda positiva significativa no AUC quando embaralhada (~0.01). As features de sentimento tem queda proxima de zero ou **negativa** — embaralhar mean_logit_neg, std21, Volume, e ma21 na verdade **melhora** o AUC! Isso indica que essas features estao adicionando ruido, nao sinal.

**Conclusao critica:** Com 17 anos de dados, as features de sentimento nao apenas nao ajudam — elas **atrapalham**. O modelo performaria melhor sem elas.

---

## 6. Curvas de aprendizado: mais dados nao ajudam

As curvas de aprendizado mostram:

- **XGBoost:** AUC estavel em ~0.50-0.52 independente da quantidade de dados. Plano.
- **Random Forest:** Ligeira melhoria de 0.48 para 0.53. Marginal.
- **Logistic Regression:** Sobe de ~0.50 para ~0.57 com mais dados. O modelo linear e o que mais se beneficia de mais dados, mas o AUC final ainda e fraco.

**Conclusao:** Adicionar mais anos de dados historicos nao melhora os resultados. O sinal de sentimento (se existe) e de curto prazo e local — padroes de 2009 nao informam sobre 2024.

---

## 7. Variacoes de hiperparametros

| Variacao | AUC | Obs |
|----------|-----|-----|
| TCN padrao | **0.594** | Melhor resultado |
| TCN k=2 | 0.584 | Proximo |
| TCN [32,32] | 0.583 | Proximo |
| TCN k=5 | 0.584 | Todas variacoes TCN > 0.58 |
| Transformer d=32 | 0.569 | Melhor Transformer |
| LR C=0.01 | 0.567 | Mais regularizado |
| Transformer 4L | 0.564 | Mais camadas nao ajudou |
| Transformer d=128 | 0.544 | Modelo maior pior |
| LR L1 | 0.532 | Similar ao padrao |
| XGBoost depth=6 | 0.506 | Marginal |
| RF depth=20 | 0.503 | Aleatorio |
| BiLSTM drop=0.5 | 0.514 | Mais dropout nao ajudou |
| BiLSTM h=256 | 0.490 | Mais neuronios piorou |
| XGBoost depth=3 | **0.489** | Pior que aleatorio |
| RF depth=5 | 0.493 | Pior que aleatorio |

**Destaque: TCN domina o Stage 6.** Todas as variacoes de TCN ficaram acima de 0.58 — bem acima dos outros modelos. Isso sugere que o TCN captura padroes locais que outros modelos perdem, mesmo com dados ruidosos de 17 anos.

Nenhuma variacao atingiu AUC > 0.60. O teto de desempenho deste dataset e muito baixo.

---

## 8. Comparacao completa: todos os stages

| | Stage 3 (Ollama) | Stage 4 (FinBERT 4y) | Stage 5b (h=5, eng.) | Stage 6 (FinBERT 17y) |
|---|:---:|:---:|:---:|:---:|
| Features | 43 (PCA) | 16 | 32 (engenheiradas) | 16 |
| Sequencias | 1.176 | 2.717 | 1.172 | 4.125 |
| Balance | 58.5/41.5 | 56.7/43.3 | 59.9/40.1 | 65.5/34.5 |
| Horizonte | 21 dias | 21 dias | **5 dias** | 21 dias |
| Melhor AUC (padrao) | XGB 0.610 | RF 0.559 | **Transf. 0.608** | TCN 0.594 |
| Melhor AUC (variacao) | Transf 4L 0.688 | RF 500 0.562 | **TCN [32,32] 0.643** | TCN 0.594 |
| Modelos prevendo 2 classes | 1 | 1 | **3** | 2 |
| Feature mais importante | std21 | ma21 | **sentimento eng.** | Close |
| Sentimento importa? | Nao | Pouco | **Sim** | **Nao (atrapalha)** |

---

## 9. Conclusoes para o TCC

### Achado 1: Mais dados historicos PIORAM os resultados

O Stage 6 (4.125 sequencias) performou pior que o Stage 4 (2.717 sequencias) e Stage 5b (1.172 sequencias). Isso confirma a hipotese de **concept drift**: a relacao entre sentimento de noticias e preco de acoes muda ao longo dos anos. Padroes de 2009-2015 confundem os modelos quando testados em 2023-2026.

### Achado 2: Sentimento historico e ruido

A permutation importance mostra que features de sentimento nao apenas nao ajudam — elas atrapalham. Embaralhar o sentimento melhora o AUC. Isso acontece porque o sentimento de 2009 tem uma relacao diferente com o preco do que o sentimento de 2023. O modelo tenta aprender uma relacao "media" que nao existe de forma consistente.

### Achado 3: TCN e o modelo mais robusto com dados ruidosos

O TCN (Temporal Convolutional Network) dominou o Stage 6 (AUC 0.594, todas variacoes > 0.58). Enquanto LSTM e Transformer tentam aprender dependencias de longo prazo (que sao ruidosas em 17 anos), o TCN foca em padroes locais (dias adjacentes) — mais robusto a mudancas de regime.

### Achado 4: Balance mais desigual = mais colapso

Com 65.5% sobe / 34.5% desce, este e o dataset mais desbalanceado. Mesmo com class weights, o colapso foi quase total. Isso sugere que class weights compensam desbalanceamentos leves (~58/42), mas sao insuficientes para desbalanceamentos maiores (~65/35) quando o sinal e fraco.

### Achado 5: Periodo de teste e o fator determinante

A estabilidade temporal mostra oscilacao extrema — AUC variando de 0.0 a 1.0 entre trimestres. Qualquer AUC reportado e uma media que esconde desempenho inconsistente. Isso reforça que resultados de previsao financeira devem ser interpretados com cautela e sempre reportados com analise de estabilidade temporal.

### Implicacao final

O Stage 5b (horizonte curto + features engenheiradas) permanece como a melhor configuracao do trabalho. A narrativa do TCC se consolida: **sentimento financeiro especifico (FinBERT), processado com features engenheiradas (medias moveis, deltas), em horizonte curto (5 dias), e a combinacao mais eficaz — nao mais dados, e sim dados mais relevantes.**
