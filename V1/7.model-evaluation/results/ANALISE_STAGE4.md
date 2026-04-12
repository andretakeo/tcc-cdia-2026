# Analise Aprofundada — Stage 4: FinBERT Sentiment (4 anos)

## Contexto

Nesta configuracao, as noticias foram processadas pelo FinBERT-PT-BR, um modelo de linguagem treinado especificamente em textos financeiros brasileiros. Em vez de gerar vetores de 1.024 dimensoes (como o Ollama no Stage 3), o FinBERT responde uma pergunta simples: "esta noticia e positiva, negativa ou neutra para o mercado?" Isso gera apenas 5 features de sentimento (n_articles, mean_logit_pos, mean_logit_neg, mean_logit_neu, mean_sentiment), combinadas com 11 features de preco, totalizando 16 features.

**Dataset:** 2.717 sequencias de 30 dias, balance 56.7% sobe / 43.3% desce (dados de ~2021 a ~2026). O balance e ligeiramente diferente do Stage 3 (58.5/41.5) porque o Stage 4 baixa precos ao vivo do Yahoo Finance (cobrindo um periodo maior), enquanto o Stage 3 usa um CSV fixo. O target ("preco sobe nos proximos 21 dias?") depende do periodo coberto — mais dias de mercado mudam a proporcao. Em ambos os casos o desbalanceamento e leve e compensado por class weights.

**Expectativa:** No Stage 4 original (sem esta avaliacao sistematica), o Transformer atingiu AUC 0.709 — o melhor resultado do trabalho. A pergunta e se isso se reproduz.

---

## 1. Resultado geral: desempenho abaixo do esperado

Os resultados desta rodada sao **significativamente piores** que os do Stage 4 original:

| Modelo | AUC (Stage 4 original) | AUC (Stage 7 re-treino) | Diferenca |
|--------|:----------------------:|:-----------------------:|:---------:|
| Transformer | **0.709** | 0.474 | -0.235 |
| XGBoost | 0.670 | 0.481 | -0.189 |
| BiLSTM Original | 0.500 | 0.500 | 0.000 |
| BiLSTM Reduzido | 0.477 | 0.519 | +0.042 |
| Random Forest | — | 0.559 | novo |
| TCN | — | 0.526 | novo |
| Logistic Regression | — | 0.505 | novo |

### Por que os resultados pioraram?

Existem tres explicacoes provaveis:

1. **Dados de preco atualizados:** O data loader usa `MarketData("ITUB4.SA").features()` que baixa dados ao vivo do Yahoo Finance. Desde a ultima execucao, novos dados foram adicionados ao periodo de teste, potencialmente incluindo um regime de mercado diferente.

2. **Split temporal diferente:** Com mais dados disponiveis, os periodos exatos de treino/validacao/teste mudaram. O periodo de teste agora cobre datas mais recentes que podem ter padroes diferentes.

3. **Class weights:** Adicionamos balanceamento de classes (pos_weight na BCELoss). Isso muda o comportamento de treino — o modelo e penalizado mais por errar "desce", o que pode ter alterado o ponto de convergencia. No Stage 4 original, os modelos PyTorch nao tinham class weights.

**Nota importante:** Isso nao invalida o resultado original. Significa que o resultado de AUC 0.709 era especifico para aquele split temporal e aquela configuracao exata. A reproducibilidade e uma questao relevante para o TCC.

---

## 2. Tabela comparativa dos 7 modelos

| Modelo | AUC | Accuracy | F1 (Sobe) | F1 (Desce) | ECE | Brier |
|--------|-----|----------|-----------|------------|-----|-------|
| Random Forest | **0.559** | 36.6% | 0.015 | 0.532 | 0.387 | 0.383 |
| TCN | 0.526 | 36.4% | 0.000 | 0.534 | 0.548 | 0.532 |
| BiLSTM Reduzido | 0.519 | 36.4% | 0.000 | 0.534 | 0.327 | 0.338 |
| Logistic Regression | 0.505 | 36.3% | 0.008 | 0.531 | 0.435 | 0.424 |
| BiLSTM Original | 0.500 | 36.4% | 0.000 | 0.534 | 0.171 | 0.261 |
| XGBoost | 0.481 | 55.9% | **0.688** | 0.248 | **0.139** | **0.250** |
| Transformer | 0.474 | 36.4% | 0.000 | 0.534 | 0.354 | 0.357 |

### Observacoes:

- **XGBoost e o unico que prevê ambas as classes** (F1 sobe = 0.688), mas com AUC abaixo de 0.5, significando que sua ordenacao e pior que aleatoria. Ele "acerta" muitas previsoes de "sobe" porque preve "sobe" com frequencia, mas nao porque realmente aprendeu.

- **Todos os outros modelos colapsaram** (F1 sobe = 0), prevendo sempre "desce". Mesmo com class weights.

- **BiLSTM Original tem o melhor ECE (0.171)** entre os modelos que colapsaram — ironicamente porque suas probabilidades (todas ~0.47-0.49) estao perto do valor real de base rate.

---

## 3. Matrizes de confusao: colapso visual

As matrizes confirmam:
- **BiLSTM, Transformer, TCN:** 149 "desce" corretos, 0 "sobe" previstos. Colapso total.
- **XGBoost:** O unico com previsoes distribuidas — 30 desce corretos, 119 desce errados como sobe, 63 sobe errados como desce, 201 sobe corretos. Mas com AUC < 0.5, estas previsoes nao sao confiaveis.
- **Logistic Regression:** Previu 1 unica amostra como "sobe" (e acertou — precision 100%, mas recall 0.4%).
- **Random Forest:** Previu 2 amostras como "sobe".

---

## 4. Distribuicao de previsoes: por que tudo colapsa

Os histogramas revelam o problema central:

- **BiLSTM Original:** Todas as probabilidades entre 0.46 e 0.50. Uma faixa de apenas 0.04 — o modelo nao consegue discriminar.

- **Transformer:** Todas entre 0.28 e 0.50. Ligeiramente mais espalhado mas ainda abaixo do threshold.

- **TCN:** Concentrado entre 0.05 e 0.30. O TCN esta extremamente pessimista — acha que quase tudo vai descer. Isso explica seu ECE alto (0.548): quando diz 10% de chance de subir, na realidade sobe ~57% das vezes.

- **XGBoost:** Probabilidades extremamente comprimidas entre 0.497 e 0.503. Uma faixa de 0.006! O modelo esta quase "jogando moeda", mas como algumas previsoes caem acima de 0.5, ele e o unico que prevê "sobe".

- **Logistic Regression:** Quase tudo entre 0.0 e 0.1. Extremamente pessimista.

- **Random Forest:** Distribuicao mais saudavel (0.20 a 0.50), com alguma separacao entre classes, mas insuficiente.

---

## 5. Estabilidade temporal: alta volatilidade

O grafico de AUC por janela de 3 meses e o mais revelador:

- **Oscilacao extrema:** Todos os modelos oscilam entre AUC 0.0 e 0.9 ao longo do tempo. Nao ha consistencia.

- **Periodos bons existem:** Ha janelas onde modelos atingem AUC > 0.8 (jul-set 2024 para alguns modelos). Mas em outras janelas caem para AUC < 0.2 (pior que aleatorio invertido).

- **Nenhum modelo domina consistentemente:** O "melhor" modelo muda a cada trimestre.

**O que isso significa:** O mercado financeiro tem regimes que mudam ao longo do tempo. Um modelo treinado em dados de 2021-2024 pode funcionar bem em um trimestre e falhar no seguinte. Isso e chamado de "concept drift" — a relacao entre sentimento e preco muda conforme o contexto macroeconomico (juros, inflacao, crises, eleicoes).

---

## 6. SHAP e Importancia de Features

### XGBoost — SHAP
As features mais impactantes sao:
1. **ma21** (media movel 21 dias) — de longe a mais importante
2. **ma7** (media movel 7 dias)
3. **lag_5** (preco de 5 dias atras)
4. **Volume**
5. **Close**

As features de sentimento (mean_sentiment, mean_logit_neu, mean_logit_pos, mean_logit_neg) aparecem no meio do ranking com impacto moderado. **mean_sentiment** e **mean_logit_neu** tem alguma contribuicao, mas bem menor que as medias moveis.

### XGBoost — Permutation Importance
**Volume** e a unica feature com queda significativa no AUC quando embaralhada. As features de sentimento tem queda proxima de zero — embaralhar o sentimento nao piora o modelo. Isso indica que o XGBoost nao esta usando sentimento de forma significativa nesta rodada.

### Logistic Regression — SHAP
**ma21** domina completamente, seguida por lags de preco. Features de sentimento (mean_sentiment, mean_logit_pos, mean_logit_neg) tem impacto negligenciavel. O modelo linear encontra relacao apenas com tendencia de preco, nao com sentimento.

---

## 7. Curvas de aprendizado: saturacao precoce

Os 3 modelos tabulares mostram curvas planas:
- **XGBoost:** AUC estavel em ~0.48 independente da quantidade de dados.
- **Random Forest:** AUC sobe de ~0.50 para ~0.56 com 100% dos dados. Leve melhoria.
- **Logistic Regression:** Estavel em ~0.50 (aleatorio).

**Conclusao:** Mais dados de treino nao ajudariam. O sinal (se existe) ja foi capturado com 20% dos dados.

---

## 8. Variacoes de hiperparametros

| Melhor variacao | AUC | Obs |
|-----------------|-----|-----|
| RF 500 trees | **0.562** | Melhor geral, marginal |
| RF depth=20 | 0.557 | Similar |
| TCN k=2 | 0.549 | Melhor entre redes neurais |
| XGBoost depth=6 | 0.539 | Melhora com mais profundidade |
| Transformer d=128 | 0.538 | Melhor Transformer |
| BiLSTM drop=0.5 | 0.537 | Mais dropout ajudou levemente |
| Transformer 4L | **0.381** | Pior resultado — mais camadas prejudicou |

Nenhuma variacao trouxe melhoria significativa. O Transformer com 4 camadas foi o pior resultado (AUC 0.381), indicando que mais complexidade prejudicou — o modelo com mais parametros overfittou os padroes de treino.

---

## 9. Comparacao Stage 3 vs Stage 4

| Metrica | Stage 3 (Ollama) | Stage 4 (FinBERT) |
|---------|:----------------:|:-----------------:|
| Melhor AUC (padrao) | XGBoost 0.610 | RF 0.559 |
| Melhor AUC (variacao) | Transformer 4L 0.688 | RF 500 trees 0.562 |
| Modelos que preveem "sobe" | 1 (XGB depth=3) | 1 (XGBoost) |
| Feature mais importante | std21 (preco) | ma21 (preco) |
| Sentimento importa? | N/A (embeddings) | Pouco (SHAP mostra contribuicao pequena) |

**Resultado surpreendente:** Nesta rodada, o Stage 3 (embeddings Ollama) teve AUC melhor que o Stage 4 (FinBERT). Isso contradiz o resultado original (Stage 4 era muito superior). As causas provaveis sao:

1. **Periodo de teste diferente:** Os dados de preco atualizados mudaram o split temporal.
2. **Class weights:** O balanceamento mudou o comportamento de convergencia dos modelos.
3. **Aleatoriedade:** Com datasets pequenos (~400 amostras de teste), resultados podem variar significativamente entre rodadas.

---

## 10. Conclusoes para o TCC

### Achado 1: Reproducibilidade e fragil
O AUC de 0.709 do Transformer no Stage 4 original nao se reproduziu (caiu para 0.474). Isso e importante para reportar honestamente no TCC. Resultados de modelos financeiros sao sensiveis ao periodo de teste e configuracao exata.

### Achado 2: Todos os modelos colapsam
Com 16 features e ~2.700 sequencias, nenhum dos 7 modelos consegue classificar de forma util. O colapso (prever tudo como "desce") e um sintoma de sinal fraco — os dados nao contem informacao suficiente para separar as classes.

### Achado 3: Sentimento contribui pouco
O SHAP e permutation importance mostram que features de sentimento FinBERT tem impacto pequeno comparado a features de preco (medias moveis, lags, volume). O sinal de sentimento, se existe, e fraco demais para ser capturado de forma robusta.

### Achado 4: Estabilidade temporal e o problema central
O grafico de estabilidade temporal revela que o desempenho oscila dramaticamente ao longo do tempo. Qualquer AUC reportado e uma media que esconde periodos de otimo desempenho e periodos de fracasso total. Isso aponta para concept drift — a relacao sentimento-preco muda com o regime de mercado.

### Implicacao para o TCC
Estes resultados sao **mais realistas** que os do Stage 4 original. Prever direcao de preco de acoes e um problema notoriamente dificil (hipotese de mercados eficientes). AUCs entre 0.50 e 0.56 sao tipicos na literatura academica. O resultado original de 0.709 provavelmente era inflado por um split temporal favoravel.
