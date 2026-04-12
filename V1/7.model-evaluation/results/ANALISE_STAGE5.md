# Analise Aprofundada — Stage 5b: Horizonte 5 dias + Features Engenheiradas

## Contexto

Este estagio testa duas mudancas simultaneas em relacao ao Stage 4:

1. **Horizonte reduzido:** 21 dias → 5 dias uteis. A hipotese era que sentimento de noticias tem impacto de curto prazo — uma noticia negativa hoje deveria afetar o preco nos proximos dias, nao daqui a um mes.

2. **Features engenheiradas de sentimento:** Alem das 5 features base do FinBERT, foram criadas 16 features derivadas: medias moveis de sentimento (7 e 21 dias), deltas (variacao do sentimento vs media), razao positivo/negativo, volume acumulado de noticias, e volatilidade de sentimento. Total: 32 features (11 preco + 5 sentimento base + 16 engenheiradas).

**Dataset:** 1.172 sequencias de 30 dias, balance 59.9% sobe / 40.1% desce. O balance e ligeiramente diferente dos outros stages porque o horizonte de 5 dias muda a proporcao de dias "sobe" vs "desce" (em periodos curtos, a tendencia de alta do mercado pesa menos). Todos os modelos usam class weights para compensar.

**Expectativa original (Stage 5b):** Horizonte mais curto + mais features de sentimento → melhor resultado. O resultado original mostrou o oposto (AUC caiu de 0.709 para 0.588 no Transformer).

---

## 1. Resultado geral: melhor que Stage 4, pior que esperado

| Modelo | AUC | Accuracy | F1 (Sobe) | F1 (Desce) | ECE | Brier |
|--------|-----|----------|-----------|------------|-----|-------|
| Transformer | **0.608** | 40.1% | 0.000 | 0.573 | 0.156 | 0.262 |
| TCN | 0.569 | 40.1% | 0.000 | 0.573 | 0.223 | 0.267 |
| Logistic Regression | 0.532 | 38.7% | 0.083 | 0.539 | 0.290 | 0.321 |
| XGBoost | 0.528 | 55.8% | **0.658** | 0.375 | **0.092** | **0.250** |
| BiLSTM Reduzido | 0.522 | 42.4% | 0.089 | 0.579 | 0.124 | 0.255 |
| Random Forest | 0.452 | 43.6% | 0.370 | 0.490 | 0.157 | 0.283 |
| BiLSTM Original | 0.434 | 40.1% | 0.000 | 0.573 | 0.121 | 0.256 |

### Comparacao com Stage 4:

| Modelo | AUC Stage 4 | AUC Stage 5b | Diferenca |
|--------|:-----------:|:------------:|:---------:|
| Transformer | 0.474 | **0.608** | +0.134 |
| TCN | 0.526 | 0.569 | +0.043 |
| XGBoost | 0.481 | 0.528 | +0.047 |
| LR | 0.505 | 0.532 | +0.027 |
| BiLSTM Red. | 0.519 | 0.522 | +0.003 |
| RF | 0.559 | 0.452 | -0.107 |
| BiLSTM Orig. | 0.500 | 0.434 | -0.066 |

**Achado:** O Transformer melhorou significativamente (+0.134 AUC) com horizonte de 5 dias. Isso sugere que o Transformer captura melhor relacoes de curto prazo entre sentimento e preco. Porem, a maioria dos modelos ainda colapsa.

---

## 2. Matrizes de confusao: mais modelos preveem ambas as classes

Comparado aos Stages 3 e 4 (onde quase tudo colapsava), o Stage 5b mostra mais diversidade:

- **BiLSTM Original, Transformer, TCN:** Ainda colapsam (tudo "desce").
- **XGBoost:** Prevê ambas as classes (77 sobe corretos, 50 desce como sobe). O unico com F1 sobe razoavel (0.658).
- **Random Forest:** Tambem prevê ambas (30 sobe corretos, 25 desce como sobe). Mais equilibrado que XGBoost.
- **Logistic Regression:** Prevê algumas amostras como "sobe" (5), com recall muito baixo.
- **BiLSTM Reduzido:** Prevê 1 amostra como "sobe" (e acertou — precision 83%).

**Progresso vs Stage 3/4:** O horizonte de 5 dias e as features engenheiradas ajudaram XGBoost e Random Forest a nao colapsar. Isso indica que as features derivadas de sentimento (medias moveis, deltas, razoes) contem mais sinal para curto prazo do que as features base sozinhas.

---

## 3. Distribuicao de previsoes: TCN e Random Forest mais saudaveis

- **BiLSTM Original:** Probabilidades entre 0.465 e 0.500. Faixa de 0.035 — colapso por indecisao.

- **Transformer:** Concentrado entre 0.42 e 0.48. Abaixo do threshold, mas com alguma dispersao.

- **TCN:** Distribuicao entre 0.38 e 0.50, com alguma separacao entre as classes verde (sobe) e vermelha (desce). Melhor que Stage 4.

- **XGBoost:** Probabilidades entre 0.48 e 0.51 — faixa de 0.03. Extremamente comprimido, mas como a distribuicao cruza o threshold de 0.5, ele consegue prever ambas as classes.

- **Random Forest:** A distribuicao mais saudavel — probabilidades entre 0.2 e 0.7, com separacao visivel entre classes. A classe "sobe" (verde) tende a ter probabilidades mais altas. Isso e o comportamento desejado, mesmo que o AUC geral seja baixo (0.452).

- **Logistic Regression:** Espalhado entre 0.0 e 0.6, mas concentrado em valores baixos.

---

## 4. Estabilidade temporal: Transformer mais estavel

O grafico de AUC por janela de 3 meses mostra:

- **Transformer (verde):** O mais estavel — mantem AUC entre 0.5 e 0.7 na maioria das janelas. Nao tem os picos extremos nem quedas abruptas dos outros modelos.

- **XGBoost (laranja):** Instavel — oscila entre 0.3 e 0.8.

- **Random Forest (verde escuro):** Cai progressivamente ao longo do tempo, sugerindo que os padroes que aprendeu estao "envelhecendo".

- **TCN (roxo):** Bom no inicio (~0.7), cai para ~0.4 no final. Comportamento similar ao RF.

**Conclusao:** O Transformer com horizonte de 5 dias e o modelo mais robusto temporalmente. Isso e relevante porque estabilidade e tao importante quanto AUC medio — um modelo que funciona bem em 3 meses e falha nos 3 seguintes nao e util na pratica.

---

## 5. SHAP e Importancia de Features: sentimento engenheirado importa!

### XGBoost — SHAP
**Achado mais importante do Stage 5b:** As features engenheiradas de sentimento dominam o ranking do SHAP:

1. **mean_logit_neu_ma7** — media movel 7 dias do logit neutro
2. **mean_logit_neg_ma21** — media movel 21 dias do logit negativo
3. **n_articles_sum21** — volume acumulado de noticias em 21 dias
4. **std21** — volatilidade de preco
5. **mean_sentiment_ma7** — media movel 7 dias do sentimento medio

Pela primeira vez no trabalho, **features de sentimento aparecem no topo do ranking de importancia**. As medias moveis de sentimento (ma7, ma21) sao mais uteis que os valores diarios brutos. Isso faz sentido: o sentimento de um unico dia e ruidoso, mas a tendencia de sentimento ao longo de uma semana captura algo mais significativo.

### XGBoost — Permutation Importance
Confirma o SHAP:
- **n_articles_sum21** e a feature com maior queda no AUC quando embaralhada
- **mean_logit_neg_ma21** e a segunda mais importante
- **std21** (preco) aparece em terceiro

**Pela primeira vez, features de sentimento superam features de preco em importancia.** Nos Stages 3 e 4, as features de preco (std21, ma7, ma21) dominavam. Aqui, as features engenheiradas de sentimento tomaram a lideranca.

### Logistic Regression — SHAP
Para o modelo linear, as features mais importantes sao todas de preco (lag_3, ma21, lag_1, Close). Features de sentimento aparecem no meio do ranking. Isso indica que a relacao sentimento-preco e **nao-linear** — modelos lineares nao a capturam, mas modelos baseados em arvores (XGBoost) sim.

---

## 6. Curvas de aprendizado

- **XGBoost:** AUC ligeiramente crescente (0.48 → 0.53) com mais dados. Leve beneficio.
- **Random Forest:** AUC estavel em ~0.45-0.50. Mais dados nao ajudam.
- **Logistic Regression:** Crescente de 0.45 para 0.53. Surpreendentemente, mais dados ajudam o modelo linear — talvez as features engenheiradas tenham relacao linear sutil que so emerge com mais amostras.

---

## 7. Variacoes de hiperparametros

| Variacao | AUC | Obs |
|----------|-----|-----|
| **TCN [32,32]** | **0.643** | Melhor resultado! F1 sobe = 0.456, previu ambas classes |
| TCN k=5 | 0.638 | Muito bom tambem |
| Transformer 4L | 0.631 | Mais camadas ajudaram (contrario do Stage 4) |
| Transformer d=32 | 0.631 | Modelo menor performou igual |
| Transformer d=128 | 0.629 | Modelo maior tambem |
| TCN k=2 | 0.626 | Todas variacoes de TCN > 0.62 |
| LR C=0.01 | 0.571 | Mais regularizacao ajudou |
| XGBoost depth=6 | 0.521 | Similar ao padrao |

### Destaque: TCN [32,32] com AUC 0.643

A melhor configuracao do Stage 5b e um TCN com apenas 2 camadas de 32 filtros (mais simples que o padrao de 3x64). Este modelo:
- AUC: 0.643 (melhor do Stage 5b)
- F1 sobe: 0.456 (previu 32% dos dias que subiram corretamente)
- Precision sobe: 0.791 (quando previu "sobe", acertou 79% das vezes)
- F1 desce: 0.605

Isso indica que para horizonte de 5 dias com features engenheiradas, um **TCN pequeno** e a melhor opcao — captura padroes convolucionais locais sem overfitting.

---

## 8. Comparacao entre todos os stages ate agora

| | Stage 3 (Ollama) | Stage 4 (FinBERT) | Stage 5b (h=5, engenheirado) |
|---|:---:|:---:|:---:|
| Melhor AUC padrao | XGBoost 0.610 | RF 0.559 | **Transformer 0.608** |
| Melhor AUC variacao | Transformer 4L 0.688 | RF 500 0.562 | **TCN [32,32] 0.643** |
| Sentimento importa? | Nao | Pouco | **Sim** (features engenheiradas) |
| Modelos que preveem 2 classes | 1 | 1 | **3** (XGB, RF, TCN [32,32]) |
| Estabilidade temporal | Ruim | Ruim | **Razoavel** (Transformer) |

---

## 9. Conclusoes para o TCC

### Achado 1: Features engenheiradas de sentimento funcionam
As medias moveis, deltas e acumulados de sentimento sao mais uteis que os valores diarios brutos. Isso mostra que a **tendencia de sentimento** (como o sentimento muda ao longo de dias/semanas) e mais informativa que o sentimento pontual.

### Achado 2: Horizonte de 5 dias favorece TCN
O TCN com configuracao simples (2 camadas, 32 filtros) atingiu o melhor resultado (AUC 0.643). Isso sugere que para previsao de curto prazo, padroes convolucionais locais (dias adjacentes) sao mais relevantes que memoria de longo prazo (LSTM) ou atencao global (Transformer).

### Achado 3: Relacao sentimento-preco e nao-linear
Logistic Regression (linear) nao consegue capturar a importancia do sentimento, mas XGBoost (nao-linear) sim. Isso indica que o efeito do sentimento sobre o preco nao e direto ("sentimento positivo → preco sobe"), mas condicional — depende de outras variaveis como volatilidade, tendencia de preco, e volume de noticias.

### Achado 4: Mais features nao significa mais colapso
Apesar de ter o dobro de features do Stage 4 (32 vs 16), o Stage 5b teve menos colapso. As features engenheiradas sao mais informativas que as brutas, dando aos modelos mais "pistas" para discriminar.

### Implicacao
A combinacao de horizonte curto + features engenheiradas de sentimento e a mais promissora ate agora. O Stage 6 (17 anos de dados) testara se mais dados historicos melhoram ou pioram estes resultados.
