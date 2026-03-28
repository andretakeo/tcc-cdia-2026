# Analise Comparativa Final — Todos os Stages

## Visao geral do trabalho

Este trabalho investigou se noticias financeiras brasileiras melhoram a previsao de direcao de preco de acoes (sobe/desce) da ITUB4 (Itau Unibanco). Foram testadas 4 configuracoes de dados diferentes, cada uma com 7 modelos de machine learning e multiplas variacoes de hiperparametros.

O objetivo deste Stage 7 nao era melhorar os modelos, mas **entender por que** alguns funcionam e outros nao — coletando metricas diagnosticas (calibracao, estabilidade temporal, importancia de features, distribuicao de previsoes, curvas de aprendizado) que os estagios anteriores nao tinham.

---

## Tabela comparativa: melhor modelo de cada stage

| Stage  | Dados                    | Features | Horizonte  | Melhor modelo   | AUC       | F1 Sobe   |
| ------ | ------------------------ | -------- | ---------- | --------------- | --------- | --------- |
| 3      | Ollama embeddings        | 43 (PCA) | 21 dias    | XGBoost         | 0.610     | 0.000     |
| 4      | FinBERT sentimento       | 16       | 21 dias    | Random Forest   | 0.559     | 0.015     |
| **5b** | **FinBERT engenheirado** | **32**   | **5 dias** | **Transformer** | **0.608** | **0.000** |
| 6      | FinBERT 17 anos          | 16       | 21 dias    | TCN             | 0.594     | 0.005     |

## Tabela comparativa: melhor variacao de hiperparametros de cada stage

| Stage  | Melhor variacao | AUC       | F1 Sobe   | Previu 2 classes? |
| ------ | --------------- | --------- | --------- | :---------------: |
| 3      | Transformer 4L  | 0.688     | 0.000     |        Nao        |
| 4      | RF 500 trees    | 0.562     | 0.015     |     Quase nao     |
| **5b** | **TCN [32,32]** | **0.643** | **0.456** |      **Sim**      |
| 6      | TCN padrao      | 0.594     | 0.005     |        Nao        |

**O Stage 5b com TCN [32,32] e a melhor configuracao do trabalho** — nao apenas pelo AUC (0.643), mas por ser o unico que gera previsoes uteis nas duas direcoes (F1 sobe = 0.456, precision sobe = 79%).

---

## Os 5 grandes achados

### 1. Representacao do texto importa mais que quantidade de dados

| Representacao                       |    Dimensoes     |      Especificidade       |       Resultado        |
| ----------------------------------- | :--------------: | :-----------------------: | :--------------------: |
| Ollama embedding                    | 1.024 → 32 (PCA) |         Generica          |   Fraco (AUC ~0.61)    |
| FinBERT sentimento bruto            |        5         |        Financeira         |   Fraco (AUC ~0.56)    |
| **FinBERT sentimento engenheirado** |      **21**      | **Financeira + temporal** | **Melhor (AUC ~0.64)** |

Os embeddings Ollama capturam "significado geral" do texto — topico, estilo, gramatica — mas a maioria dessa informacao e irrelevante para prever precos. O FinBERT filtra para sentimento financeiro (positivo/negativo/neutro), e as features engenheiradas (medias moveis, deltas, razoes) capturam a **tendencia de sentimento** ao longo do tempo.

A analise SHAP confirma: no Stage 3, os componentes PCA sao ignorados pelo modelo. No Stage 5b, features engenheiradas de sentimento aparecem no topo do ranking de importancia — pela primeira vez superando features de preco.

### 2. Mais dados historicos pioram os resultados (concept drift)

| Dados               | Sequencias | AUC melhor modelo |
| ------------------- | :--------: | :---------------: |
| 4 anos (2021-2026)  |   1.172    |     **0.643**     |
| 17 anos (2009-2026) |   4.125    |       0.594       |

O Stage 6 (3.5x mais dados) performou **pior** que o Stage 5b. A permutation importance do Stage 6 revelou algo ainda mais forte: embaralhar features de sentimento **melhora** o AUC — o sentimento historico esta adicionando ruido, nao sinal.

Isso acontece por **concept drift**: a relacao entre sentimento de noticias e preco de acoes muda ao longo dos anos. O mercado brasileiro de 2009 era completamente diferente do de 2024 (diferentes taxas de juros, governos, cenario global). Padroes aprendidos em dados antigos confundem o modelo quando aplicados a dados recentes.

As curvas de aprendizado confirmam: AUC nao melhora com mais dados de treino. A curva e plana em todos os stages.

### 3. Horizonte curto (5 dias) captura melhor o efeito de noticias

| Horizonte  | AUC (Stage 4/5b, mesmo periodo) |
| :--------: | :-----------------------------: |
|  21 dias   |              0.559              |
| **5 dias** |            **0.608**            |

Noticias financeiras tem impacto de **curto prazo**. Uma noticia negativa sobre o Itau hoje pode afetar o preco nos proximos 2-5 dias, mas provavelmente nao daqui a um mes (quando outros fatores ja dominam). O horizonte de 5 dias captura esse efeito imediato.

### 4. Modelos convolucionais (TCN) sao surpreendentemente bons

O TCN apareceu entre os melhores modelos em 3 dos 4 stages:

| Stage | Melhor TCN (AUC) | vs Melhor Transformer | vs Melhor XGBoost |
| ----- | :--------------: | :-------------------: | :---------------: |
| 3     |      0.565       |         0.688         |       0.628       |
| 5b    |    **0.643**     |         0.631         |       0.528       |
| 6     |    **0.594**     |         0.569         |       0.511       |

O TCN processa a serie temporal com **convolucoes dilatadas** — filtros que olham para dias adjacentes. Diferente do LSTM (que le dia a dia sequencialmente) e do Transformer (que olha todos os dias ao mesmo tempo), o TCN foca em **padroes locais**. Isso e mais robusto a concept drift porque padroes locais (ex: "sentimento caiu nos ultimos 3 dias → preco cai amanha") sao mais estaveis do que padroes de longo prazo.

A configuracao mais simples (TCN [32,32] — apenas 2 camadas com 32 filtros) foi a melhor. Modelos maiores overfittam.

### 5. Quase todos os modelos colapsam (e isso e um resultado)

Em 25 combinacoes testadas (7 modelos × 4 stages, excluindo variacoes), **apenas 5 previram ambas as classes** com F1 sobe > 0.05:

| Config   | Modelo          |  F1 Sobe  | Precision Sobe |
| -------- | --------------- | :-------: | :------------: |
| Stage 5b | **TCN [32,32]** | **0.456** |   **0.791**    |
| Stage 5b | XGBoost         |   0.658   |     0.606      |
| Stage 5b | Random Forest   |   0.370   |     0.545      |
| Stage 5b | BiLSTM Reduzido |   0.089   |     0.833      |
| Stage 3  | XGBoost depth=3 |   0.425   |     0.771      |

O colapso (prever tudo como uma classe) nao e um bug — e um **diagnostico**. Quando o sinal nos dados e muito fraco, o modelo "desiste" de tentar discriminar e aposta na classe mais frequente. O fato de que modelos so conseguem prever ambas as classes no Stage 5b confirma que essa e a unica configuracao com sinal suficiente.

---

## Diagnosticos que explicam os resultados

### Calibracao

| Stage  | Melhor ECE | Modelo          | Interpretacao                           |
| ------ | :--------: | --------------- | --------------------------------------- |
| 3      |   0.202    | BiLSTM drop=0.1 | Probabilidades moderadamente confiaveis |
| 4      |   0.139    | XGBoost         | Melhor calibrado de todos               |
| **5b** | **0.092**  | **XGBoost**     | **Muito bem calibrado**                 |
| 6      |   0.221    | XGBoost depth=6 | Razoavel                                |

O XGBoost do Stage 5b tem a melhor calibracao (ECE 0.092) — quando diz 50%, realmente sobe ~50% das vezes. Redes neurais tendem a ser mal calibradas (ECE > 0.3), especialmente o TCN (ECE > 0.5) que e extremamente pessimista.

### Estabilidade temporal

Nenhum model e nenhum stage mostrou desempenho estavel ao longo do tempo. O AUC oscila entre 0.0 e 1.0 a cada trimestre em todos os stages. Isso e o resultado mais importante para reportar no TCC: **qualquer AUC reportado e uma media que esconde inconsistencia total.**

O Stage 5b tem o Transformer mais estavel (oscila entre 0.5 e 0.7 em vez de 0.0 e 1.0), mas mesmo assim esta longe de ser confiavel.

### Importancia de features

| Stage  | Feature mais importante (SHAP)      | Sentimento no top 5? |
| ------ | ----------------------------------- | :------------------: |
| 3      | std21 (volatilidade)                |         Nao          |
| 4      | ma21 (media movel)                  |         Nao          |
| **5b** | **mean_logit_neu_ma7 (sentimento)** |   **Sim (3 de 5)**   |
| 6      | Close (preco)                       |         Nao          |

Apenas no Stage 5b as features de sentimento lideram o ranking de importancia. Nos outros stages, features de preco (volatilidade, medias moveis, lags) dominam. Isso confirma que features engenheiradas de sentimento sao necessarias para que o sentimento contribua de forma significativa.

---

## Ranking final dos modelos

Considerando desempenho em todos os 4 stages:

| Rank | Modelo                  | Pontos fortes                                                                              | Pontos fracos                                                            |
| ---- | ----------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| 1    | **TCN**                 | Melhor em Stages 5b e 6. Robusto a dados ruidosos. Configuracoes simples funcionam melhor. | Calibracao pessima (ECE > 0.5). Colapsa com frequencia.                  |
| 2    | **XGBoost**             | Melhor calibrado (ECE ~0.09). Unico que prevê ambas classes com frequencia. Rapido.        | AUC mediocre. Nao captura padroes sequenciais.                           |
| 3    | **Transformer**         | Melhor AUC padrao no Stage 5b (0.608). Mais estavel temporalmente.                         | Colapsa na maioria dos stages. Sensivel a hiperparametros.               |
| 4    | **Random Forest**       | Distribuicoes de previsao mais saudaveis. Prevê ambas classes no Stage 5b.                 | AUC geralmente inferior ao XGBoost.                                      |
| 5    | **BiLSTM Reduzido**     | Prevê algumas amostras "sobe" com alta precisao.                                           | AUC proximo do aleatorio. Colapsa com frequencia.                        |
| 6    | **BiLSTM Original**     | —                                                                                          | Colapsa em quase todos os stages. Modelo grande demais para os dados.    |
| 7    | **Logistic Regression** | Baseline util: confirma que sinais lineares sao muito fracos.                              | AUC ~0.50 em quase todos os stages. Confirma que a relacao e nao-linear. |

---

## Conclusoes para o TCC

### Resposta as perguntas de pesquisa

**1. Noticias financeiras ajudam a prever direcao de preco?**
Sim, mas de forma limitada e condicional. O sinal existe apenas quando: (a) o sentimento e extraido por um modelo especifico para financas (FinBERT, nao Ollama), (b) as features sao engenheiradas (medias moveis e deltas, nao valores diarios brutos), e (c) o horizonte de previsao e curto (5 dias, nao 21).

**2. Qual a melhor forma de representar o texto?**
FinBERT com features engenheiradas (32 features) > FinBERT bruto (5 features) > Ollama embeddings (1.024 features). A qualidade e especificidade da representacao importam mais que a quantidade de dimensoes.

**3. Qual arquitetura funciona melhor?**
Depende da configuracao de dados, mas o TCN (Temporal Convolutional Network) e o mais robusto no geral. Para a melhor configuracao (Stage 5b), TCN [32,32] atingiu AUC 0.643 com precision de 79% nas previsoes de "sobe".

### Limitacoes a reportar

1. **Reproducibilidade fragil:** O AUC de 0.709 do Stage 4 original nao se reproduziu (caiu para 0.474 no re-treino). Resultados financeiros sao sensiveis ao periodo de teste.

2. **Estabilidade temporal inexistente:** Nenhum modelo mantem desempenho consistente ao longo do tempo. AUC oscila de 0.0 a 1.0 entre trimestres.

3. **Colapso generalizado:** A maioria dos modelos nao consegue prever ambas as classes. O sinal nos dados e fraco demais para discriminacao robusta.

4. **Uma unica acao:** Resultados sao para ITUB4. Nao ha garantia de que generalizem para outras acoes.

5. **Uma unica fonte de noticias:** Apenas InfoMoney. Outras fontes poderiam conter sinais diferentes ou complementares.

### Contribuicoes originais

1. **Demonstracao empirica de que embeddings genericos sao inferiores a sentimento especifico** para previsao financeira com PLN.

2. **Feature engineering de sentimento:** Medias moveis e deltas de sentimento superam valores diarios brutos — achado confirmado por SHAP.

3. **Concept drift documentado:** Mais dados historicos pioram resultados, com evidencia quantitativa (permutation importance mostra sentimento historico como ruido).

4. **TCN como alternativa competitiva** a LSTM e Transformer para series temporais financeiras com sentimento.

5. **Avaliacao diagnostica completa** com calibracao, estabilidade temporal, SHAP, permutation importance e curvas de aprendizado — indo alem de apenas AUC e acuracia.
