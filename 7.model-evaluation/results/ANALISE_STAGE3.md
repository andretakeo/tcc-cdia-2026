# Analise Aprofundada — Stage 3: Ollama Embeddings

## Contexto

Nesta configuracao, cada noticia foi transformada em um vetor de 1.024 numeros pelo modelo Ollama (embedding generico). Esses vetores foram comprimidos para 32 dimensoes via PCA, combinados com 11 features de preco (Close, Volume, retorno, medias moveis, lags), totalizando 43 features. O objetivo: prever se o preco sobe ou desce em 21 dias uteis.

**Dataset:** 1.176 sequencias de 30 dias, balance 58.5% sobe / 41.5% desce. O balance varia entre stages porque cada configuracao cobre periodos e fontes de dados diferentes. O Stage 3 usa um CSV fixo (`dataset_full.csv`, ~2021-2026), enquanto outros stages baixam precos ao vivo do Yahoo Finance. O target ("preco sobe nos proximos 21 dias uteis?") depende diretamente do periodo coberto — em periodos de alta prolongada, ha mais dias "sobe". O desbalanceamento e leve (~58/42) e esta sendo compensado por class weights em todos os modelos.

---

## 1. Resultado geral: colapso generalizado

O resultado mais importante deste estagio e negativo — e isso e um achado valido.

**Todos os 7 modelos colapsaram.** As matrizes de confusao mostram que quase todos preveem 100% "desce", ignorando completamente a classe "sobe". Mesmo com class weights balanceados (penalizando mais erros na classe minoritaria), os modelos nao conseguiram encontrar padroes uteis nos embeddings.

| Modelo | AUC | F1 (Sobe) | F1 (Desce) | Previu "sobe"? |
|--------|-----|-----------|------------|----------------|
| XGBoost | 0.610 | 0.000 | 0.471 | Nao |
| Random Forest | 0.588 | 0.000 | 0.471 | Nao |
| Transformer | 0.574 | 0.000 | 0.468 | Nao |
| BiLSTM Reduzido | 0.568 | 0.000 | 0.468 | Nao |
| Logistic Regression | 0.490 | 0.060 | 0.459 | Quase nao (4 amostras) |
| BiLSTM Original | 0.457 | 0.000 | 0.468 | Nao |
| TCN | 0.416 | 0.000 | 0.468 | Nao |

**O que isso significa:** Os embeddings Ollama de 1.024 dimensoes, mesmo reduzidos para 32 via PCA, sao essencialmente ruido para este problema. Os modelos nao conseguem separar dias que vao subir de dias que vao descer usando essas representacoes textuais genericas.

---

## 2. Curvas ROC: sinal fraco mas existente

As curvas ROC mostram que alguns modelos ficam ligeiramente acima da diagonal (aleatorio), indicando que existe um sinal muito fraco nos dados. XGBoost (AUC 0.610) e Random Forest (0.588) conseguem ordenar as previsoes um pouco melhor que o acaso, mas nao o suficiente para gerar previsoes uteis.

Ponto importante: **AUC mede ordenacao, nao classificacao.** Um modelo pode ter AUC 0.61 (ordena razoavelmente) mas F1 = 0 (nao consegue classificar). Isso acontece quando as probabilidades previstas estao todas comprimidas em uma faixa estreita — o modelo "sabe" um pouco qual e mais provavel, mas nao o suficiente para separar as classes com um threshold.

---

## 3. Distribuicao de previsoes: o diagnostico do colapso

Os histogramas de probabilidades previstas revelam exatamente por que os modelos colapsam:

- **BiLSTM Original, Reduzido, Transformer, TCN:** Todas as probabilidades ficam entre 0.38 e 0.50. O modelo nunca da probabilidade > 0.5 para nenhuma amostra, entao com threshold = 0.5, tudo e classificado como "desce". O modelo e tao incerto que suas previsoes ficam "grudadas" perto de 0.45.

- **XGBoost:** Probabilidades entre 0.35 e 0.52, com a grande maioria abaixo de 0.5. Ligeiramente mais espalhado que as redes neurais, mas ainda comprimido.

- **Logistic Regression:** Distribuicao mais ampla (0.1 a 0.6), mas a maioria abaixo de 0.5. E o unico modelo que consegue prever algumas amostras como "sobe" (4 amostras).

- **Random Forest:** Probabilidades entre 0.20 e 0.50, quase todas abaixo de 0.45. O Random Forest e naturalmente conservador (medias de muitas arvores tendem ao centro).

**Conclusao:** Os modelos nao estao "errando" — estao dizendo "nao sei". Quando um modelo da probabilidade 0.45 para tudo, ele esta admitindo que nao encontrou padroes para distinguir as classes.

---

## 4. Calibracao: probabilidades nao sao confiaveis

Os diagramas de confiabilidade mostram problemas serios:

- **BiLSTM, Transformer, TCN (ECE 0.27-0.34):** As curvas sao irregulares e longe da diagonal. As probabilidades nao refletem a realidade.

- **Logistic Regression (ECE 0.509):** A pior calibracao. Quando o modelo diz 30%, a taxa real e muito diferente de 30%.

- **XGBoost (ECE 0.251):** O melhor calibrado, mas ainda distante do ideal.

- **Random Forest (ECE 0.422):** Mal calibrado — as probabilidades de RF sao medias de votos de arvores, que tendem a ser imprecisas.

**Conclusao:** Nenhum modelo produz probabilidades confiaveis. Mesmo que voce ajustasse o threshold, as probabilidades nao sao informativas o suficiente para tomar decisoes.

---

## 5. Estabilidade temporal: desempenho inconsistente

O grafico de AUC por janela de 3 meses mostra oscilacao significativa:

- **XGBoost** (laranja): Comeca com AUC ~0.55-0.60, depois cai e se recupera. Inconsistente.
- **Random Forest** (verde): Semelhante ao XGBoost.
- **Redes neurais:** AUC oscila entre 0.3 e 0.7 sem padrao claro. Isso indica que o desempenho depende do periodo especifico do mercado, nao de padroes reais aprendidos.

**Conclusao:** Nenhum modelo manteve desempenho estavel ao longo do tempo. Os padroes capturados (se existem) sao temporarios e nao generalizaveis.

---

## 6. SHAP e Importancia de Features: embeddings sao irrelevantes

### XGBoost — SHAP
As features mais importantes para o XGBoost sao **todas de preco**, nao de embedding:
1. `std21` (volatilidade) — de longe a mais importante
2. `ma7` (media movel 7 dias)
3. `lag_1` (preco do dia anterior)
4. `ma21` (media movel 21 dias)
5. `Close` (preco de fechamento)

Os componentes PCA (pca_0 a pca_31) aparecem com importancia minima e dispersa. O modelo praticamente ignora os embeddings e se apoia apenas nas features de preco.

### XGBoost — Permutation Importance
Confirma o SHAP: `std21` e a unica feature com queda significativa no AUC quando embaralhada (queda de ~0.13). Embaralhar qualquer componente PCA individual nao afeta o AUC, confirmando que os embeddings nao contribuem.

### Logistic Regression — SHAP
As features mais importantes sao `lag_4`, `Close`, `ma21`, `lag_5`, `std21`. Novamente, apenas features de preco. Os componentes PCA tem impacto proximo de zero.

### Random Forest — SHAP
Mostra apenas Volume e Close como relevantes, com impacto SHAP muito pequeno (~0.05). Os embeddings nem aparecem.

---

## 7. Curvas de aprendizado: mais dados nao ajudariam

As curvas de aprendizado dos 3 modelos tabulares mostram:

- **XGBoost:** AUC estavel em ~0.60-0.62 independente da quantidade de dados. A curva e plana — mais dados nao melhoram.
- **Random Forest:** Semelhante, estavel em ~0.55-0.60.
- **Logistic Regression:** Estavel em ~0.50 (aleatorio). Nao ha sinal linear nos dados.

**Conclusao:** O problema nao e falta de dados. O problema e que os embeddings Ollama nao contem informacao util para prever direcao de preco. Adicionar mais artigos com embeddings Ollama nao melhoraria os resultados.

---

## 8. Variacoes de hiperparametros: nenhuma melhora significativa

| Melhor variacao | AUC | Obs |
|-----------------|-----|-----|
| Transformer 4L | 0.688 | Melhor AUC, mas ainda colapsa (F1 sobe = 0) |
| RF depth=5 | 0.658 | Arvores rasas ordenam melhor, mas nao classificam |
| XGBoost depth=6 | 0.628 | Marginal |
| XGBoost depth=3 | 0.622 | **Unico que preveu ambas classes** (F1 sobe = 0.43, precision 0.77) |

O XGBoost com depth=3 e o unico que conseguiu prever "sobe" com alguma confianca: das vezes que previu "sobe", acertou 77% (precision alta). Porem, so identificou 29% dos dias que realmente subiram (recall baixo). Isso sugere que o modelo encontrou um padrao muito especifico nas features de preco, nao nos embeddings.

---

## 9. Conclusoes para o TCC

### Achado principal
**Embeddings genericos (Ollama) nao sao eficazes para prever direcao de preco.** Os modelos extraem sinal apenas das features de preco (especialmente volatilidade e medias moveis), ignorando completamente os 32 componentes PCA dos embeddings.

### Por que os embeddings falharam
1. **Dimensionalidade excessiva:** 1.024 dimensoes para representar sentimento e extremamente redundante. PCA reduz para 32, mas a maioria da variancia capturada nao e relevante para o problema.
2. **Representacao generica:** O modelo Ollama nao foi treinado para financas. Ele captura significado geral do texto (topico, estilo, gramatica), nao sentimento financeiro especifico.
3. **Ruido mascarando sinal:** Com 32 features PCA adicionadas a 11 de preco, o ratio sinal/ruido cai. Os modelos gastam capacidade tentando usar features irrelevantes.

### Implicacao
Este resultado motiva diretamente o Stage 4 (FinBERT): trocar 1.024 dimensoes genericas por 5 dimensoes de sentimento financeiro especifico. A hipotese e que menos features com mais significado produzem melhores resultados. Os resultados do Stage 4 confirmarao ou refutarao essa hipotese.
