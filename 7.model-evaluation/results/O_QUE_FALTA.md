# O Que Falta Para Virar um Modelo de Fato

## Situacao atual

O melhor resultado do trabalho e o TCN [32,32] no Stage 5b:
- AUC: 0.643
- Precision sobe: 79% (quando prevê "sobe", acerta 79% das vezes)
- F1 sobe: 0.456
- Configuracao: FinBERT + features engenheiradas + horizonte 5 dias

Esses numeros sao promissores, mas **nao significam que o modelo e util na pratica**. Abaixo, uma lista do que separa esses resultados de algo que poderia ser usado de verdade, com explicacao de por que cada item importa.

---

## 1. Backtest financeiro (o modelo da lucro?)

### O que e
Um backtest simula o que aconteceria se voce tivesse seguido as previsoes do modelo no passado. Funciona assim:
- Dia 1: modelo diz "sobe" → voce compra a acao
- Dia 5: modelo diz "desce" → voce vende
- Repete ate o final do periodo de teste
- No final, calcula: quanto dinheiro voce ganhou ou perdeu?

### Por que falta
AUC e precision medem se o modelo **acerta a direcao**, mas nao medem se ele **gera dinheiro**. Exemplo:

- O modelo acerta 5 dias onde o preco subiu 0.1% cada → ganho total: 0.5%
- O modelo erra 1 dia onde o preco caiu 3% → perda: 3%
- Saldo: -2.5% (prejuizo, apesar de acertar 5 de 6 vezes)

### Metricas que precisaria calcular

| Metrica | O que mede | Por que importa |
|---------|-----------|-----------------|
| **Retorno acumulado** | Quanto dinheiro a estrategia gerou no total | O numero mais basico — se e negativo, o modelo e inutil na pratica |
| **Sharpe Ratio** | Retorno ajustado pelo risco (retorno medio / volatilidade) | Um retorno de 10% com volatilidade de 5% e muito melhor que 10% com volatilidade de 30%. Sharpe > 1 e considerado bom, > 2 e excelente |
| **Maximum Drawdown** | A maior queda do pico ao vale da curva de capital | Mede o "pior cenario" — quanto voce perderia no pior momento. Drawdown de -50% significa que em algum ponto voce perdeu metade do capital |
| **Win Rate** | Percentual de operacoes lucrativas | Diferente do accuracy do modelo — aqui conta se a operacao (comprar e vender) deu lucro, nao so se a direcao estava certa |
| **Profit Factor** | Soma dos lucros / soma dos prejuizos | Acima de 1 = lucrativo. Acima de 2 = muito bom |
| **vs Buy-and-Hold** | Comparacao com simplesmente comprar e segurar a acao | Se o modelo nao bate "comprar e nao fazer nada", ele e inutil — o esforço de previsao nao compensa |

### Como implementar
Seria um novo notebook que:
1. Carrega as previsoes do melhor modelo (TCN [32,32])
2. Simula compra nos dias previstos como "sobe" e venda nos "desce"
3. Calcula todas as metricas acima
4. Compara com buy-and-hold da ITUB4 no mesmo periodo

---

## 2. O modelo acerta nos dias que importam?

### O que e
Nem todo dia "sobe" e igual. Um dia que sobe 0.1% e um que sobe 5% contam o mesmo para AUC e accuracy, mas financeiramente sao completamente diferentes. Precisamos saber: **o modelo acerta mais nos dias de grande movimento ou nos pequenos?**

### Por que importa
Se o modelo so acerta dias de movimentos pequenos (±0.1%) e erra os dias de movimentos grandes (±3%), ele nao vale nada na pratica — o lucro dos acertos pequenos nao compensa o prejuizo dos erros grandes.

### O que medir
- Retorno medio dos dias previstos como "sobe" vs retorno medio dos dias previstos como "desce"
- Correlacao entre a probabilidade prevista e o tamanho do movimento real
- AUC separado para dias de alto movimento (top 25%) vs dias de baixo movimento (bottom 25%)

---

## 3. Retreino periodico (rolling window)

### O que e
Em vez de treinar o modelo uma unica vez com dados fixos e testar no futuro, o modelo seria **retreinado periodicamente** (ex: a cada mes) usando os dados mais recentes.

### Por que falta
O diagnostico de estabilidade temporal mostrou que o AUC oscila de 0.0 a 1.0 entre trimestres. Isso acontece porque a relacao sentimento-preco muda ao longo do tempo (concept drift). Um modelo treinado com dados de jan-jun pode funcionar bem em jul-set mas falhar em out-dez.

### Como funcionaria
```
Mes 1: Treinar com meses 1-12, testar no mes 13
Mes 2: Treinar com meses 2-13, testar no mes 14
Mes 3: Treinar com meses 3-14, testar no mes 15
...
```

Cada vez, o modelo "esquece" os dados mais antigos e aprende os mais recentes. Isso acompanha as mudancas do mercado.

### Impacto esperado
- AUC mais estavel ao longo do tempo (menos oscilacao)
- Resultados mais realistas (cada previsao so usa dados do passado, nunca do futuro)
- Custo: mais tempo de computacao (retreinar N vezes em vez de 1)

---

## 4. Walk-forward validation

### O que e
O split fixo 70/15/15 e uma unica "foto" do desempenho. O modelo pode ter tido sorte (ou azar) com aquele periodo especifico de teste. Walk-forward validation repete o processo em multiplas janelas:

```
Rodada 1: Treino [jan-dez 2021], Teste [jan-mar 2022]
Rodada 2: Treino [abr 2021 - mar 2022], Teste [abr-jun 2022]
Rodada 3: Treino [jul 2021 - jun 2022], Teste [jul-set 2022]
...
```

### Por que falta
Com um unico split, o AUC de 0.643 pode ser especifico daquele periodo. Walk-forward daria:
- AUC medio ± desvio padrao (ex: 0.58 ± 0.12)
- Numero de janelas onde o modelo supera o aleatorio
- Confianca estatistica de que o modelo realmente funciona

### Diferenca do split fixo
- Split fixo: "o modelo teve AUC 0.643 neste periodo especifico"
- Walk-forward: "o modelo teve AUC 0.58 ± 0.12 em media ao longo de 10 periodos diferentes, superando o aleatorio em 7 de 10"

A segunda afirmacao e muito mais forte para um TCC.

---

## 5. Mais fontes de dados

### O que temos
Apenas noticias do InfoMoney, processadas por FinBERT-PT-BR.

### O que falta

| Fonte | O que traria | Dificuldade |
|-------|-------------|:-----------:|
| **Outras midias** (Valor Economico, Bloomberg BR) | Mais cobertura, menos vies de uma unica fonte | Media |
| **Twitter/X** | Sentimento em tempo real, reacao imediata do mercado | Alta (API, volume, limpeza) |
| **Fatos relevantes (CVM)** | Comunicados oficiais do Itau (resultados, dividendos) | Baixa |
| **Dados macroeconomicos** | Selic, IPCA, cambio, risco-pais | Baixa |
| **Dados de mercado extras** | Order flow, bid-ask spread, opcoes | Alta |

### Impacto esperado
Cada fonte adicional poderia capturar um aspecto diferente que as noticias do InfoMoney sozinhas nao cobrem. Porem, mais features tambem significam mais risco de overfitting — a engenharia de features precisa ser cuidadosa.

---

## 6. Calibracao do TCN

### O problema
O TCN tem o melhor AUC (0.643) mas a **pior calibracao** (ECE > 0.5). Isso significa:
- Quando o TCN diz "10% de chance de subir", na realidade sobe ~57% das vezes
- Quando diz "30% de chance de subir", tambem sobe ~57%
- As probabilidades nao sao confiaveis — nao da para usar como "confianca" da previsao

### A solucao: pos-calibracao

Apos treinar o modelo, aplicar uma tecnica de calibracao nos outputs:

| Tecnica | Como funciona | Quando usar |
|---------|--------------|-------------|
| **Platt Scaling** | Ajusta uma sigmoid nos logits usando os dados de validacao | Quando a relacao probabilidade-realidade e monotonica |
| **Isotonic Regression** | Ajusta uma funcao nao-parametrica | Quando a relacao e irregular |
| **Temperature Scaling** | Divide os logits por uma constante (temperatura) | Quando o modelo e sistematicamente super/subconfiante |

### Impacto
- Nao muda o AUC (a ordenacao das previsoes permanece igual)
- Melhora o ECE (as probabilidades passam a refletir a realidade)
- Permite usar o modelo com thresholds confiaveis (ex: "so opero quando a probabilidade e > 70%")

---

## 7. Ensemble de modelos

### O que e
Em vez de usar um unico modelo, combinar as previsoes de varios modelos. Exemplo:
- TCN diz 60% sobe
- XGBoost diz 55% sobe
- Transformer diz 45% sobe
- Media: 53.3% sobe → prevê "sobe"

### Por que ajudaria
Cada modelo captura padroes diferentes:
- TCN: padroes locais (dias adjacentes)
- XGBoost: relacoes nao-lineares entre features
- Transformer: relacoes de atencao entre dias distantes

Combinar pode cancelar erros individuais e produzir previsoes mais robustas.

### Como implementar
- **Media simples:** Media das probabilidades de N modelos
- **Media ponderada:** Modelos com melhor AUC no periodo recente recebem mais peso
- **Stacking:** Um meta-modelo (ex: Logistic Regression) aprende a combinar as previsoes dos modelos base

---

## Resumo: prioridade de implementacao

Se voce fosse continuar o trabalho alem do TCC, esta seria a ordem de prioridade:

| Prioridade | Item | Impacto | Esforco |
|:----------:|------|:-------:|:-------:|
| 1 | Backtest financeiro | Alto | Baixo (1 notebook) |
| 2 | Walk-forward validation | Alto | Medio (refatorar treino) |
| 3 | Calibracao do TCN | Medio | Baixo (poucas linhas) |
| 4 | Retreino periodico | Alto | Medio |
| 5 | Ensemble de modelos | Medio | Baixo |
| 6 | Mais fontes de dados | Alto | Alto |
| 7 | Analise de magnitude | Medio | Baixo |

Os itens 1-3 poderiam ser implementados como um Stage 8 do TCC. Os itens 4-7 seriam trabalhos futuros.

---

## Conclusao honesta

O modelo atual (TCN [32,32], AUC 0.643) mostra que **existe sinal** na combinacao sentimento FinBERT + features engenheiradas + horizonte curto. Porem, esse sinal e:
- **Fraco:** AUC 0.643 esta longe dos 0.80+ necessarios para uso confiavel
- **Instavel:** Oscila dramaticamente entre trimestres
- **Nao validado financeiramente:** Nunca foi testado se gera lucro

Para virar um modelo "de fato", precisaria no minimo de: backtest financeiro positivo, walk-forward validation consistente, e calibracao adequada. Isso e tipico da area — a grande maioria dos modelos academicos de previsao financeira nao sobrevive ao teste da realidade.

O valor do trabalho nao esta em ter criado um modelo lucrativo (poucos conseguem), mas em ter **demonstrado empiricamente** quais combinacoes de representacao textual, features e horizontes produzem sinal, e quais nao — com diagnosticos completos que explicam por que.
