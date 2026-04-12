# Preparacao para Banca — Perguntas e Respostas

## Status: EM PROGRESSO

---

## 1. Qual e o objetivo do TCC?

Investigar se informacoes extraidas de noticias financeiras brasileiras, combinadas com dados de mercado, melhoram a previsao da **direcao do preco** (sobe ou desce) de acoes da B3.

O pipeline coleta noticias do InfoMoney, extrai representacoes numericas do texto de duas formas — embeddings genericos (Ollama) e sentimento financeiro (FinBERT-PT-BR) — combina com features de preco (fechamento, volume, medias moveis) e treina modelos de classificacao binaria para prever se o preco sobe ou desce em 21 dias uteis.

### Perguntas de pesquisa:
1. Noticias financeiras ajudam a prever direcao de preco, ou dados de preco sozinhos bastam?
2. Qual a melhor forma de representar o texto — embeddings genericos ou sentimento financeiro?
3. Qual arquitetura de modelo funciona melhor para essa tarefa?

---

## 2. Por que voce testou duas representacoes de texto e qual foi melhor?

**Sua resposta:**

Eu testei as duas porque eu precisava de uma forma de extrair dados do texto com PLN, e me imaginei que talvez uma grande quantidade de features extraidas poderia significar uma analise mais precisa. Mas por causa do desempenho tao baixo, eu percebi que tanta informacao poderia atrapalhar.

O teste usando o FinBERT foi melhor justamente por causa da falta de "lixo" (excesso de informacao), focando no que importa e ajudando o modelo a generalizar melhor.

**Ponto extra para mencionar:** O FinBERT e especifico para financas — ele entende que "queda nos juros" pode ser positivo para acoes, enquanto um modelo generico interpretaria "queda" como negativo. Isso filtra a informacao textual para o que realmente importa para o mercado.

### Conceitos-chave:
- **Embedding (Ollama):** Vetor de 1.024 numeros que representa o "significado geral" do texto. Captura tudo — estilo, topico, gramatica, sentimento — tudo misturado. E como tirar uma foto de uma pagina inteira: tem muita informacao, mas a maioria e irrelevante para prever preco de acao.
- **Sentimento (FinBERT):** 3 numeros (positivo, negativo, neutro) que respondem uma pergunta especifica: "essa noticia e boa ou ruim para o mercado?" E como ler a pagina e anotar so o que importa para investimento.

---

## 3. Por que o Transformer foi melhor que o LSTM?

**AINDA NAO PREPARADO** — precisa estudar.

Pontos que voce vai precisar explicar:
- O mecanismo de atencao permite que o Transformer "olhe" para todos os dias da janela ao mesmo tempo, enquanto o LSTM processa dia a dia sequencialmente
- O LSTM sofre com o problema do "gradiente desaparecendo" — informacoes do inicio da janela se perdem ao longo do processamento
- O Transformer nao tem esse problema porque acessa qualquer dia diretamente
- Com 16 features (poucas), o Transformer consegue focar nas relacoes temporais entre sentimento e preco sem se perder em ruido

---

## 4. Por que mais dados (17 anos) deu resultado pior que 4 anos?

**AINDA NAO PREPARADO** — precisa estudar.

Pontos que voce vai precisar explicar:
- Regime de mercado muda ao longo do tempo (crises, mudancas regulatorias, pandemia)
- Padroes de 2009-2015 podem ser completamente diferentes de 2020-2026
- O modelo aprende padroes antigos que ja nao funcionam e isso "confunde" as previsoes recentes
- O periodo de teste (ago/2023 a fev/2026) pode ter caracteristicas diferentes do treino
- Isso se chama "concept drift" — a relacao entre as features e o target muda ao longo do tempo

---

## 5. O que significam as metricas?

**AINDA NAO PREPARADO** — precisa estudar.

Metricas que voce precisa saber explicar:
- **ROC-AUC**: Capacidade de ordenar previsoes corretamente. 0.5 = aleatorio, 1.0 = perfeito
- **Acuracia**: % de acertos. Enganosa com classes desbalanceadas
- **Precisao**: Dos que o modelo disse "sobe", quantos realmente subiram?
- **Recall**: Dos que realmente subiram, quantos o modelo pegou?
- **F1**: Equilibrio entre precisao e recall
- **Matriz de confusao**: Tabela mostrando acertos e erros por classe

---

## 6. O que significa o modelo ter "colapsado"?

**AINDA NAO PREPARADO** — precisa estudar.

Varios dos seus modelos (BiLSTM, XGBoost em alguns estagios) colapsaram — previam sempre a mesma classe. Voce precisa saber explicar por que isso acontece e o que isso indica sobre o modelo.

---

## 7. Por que voce escolheu horizonte de 21 dias e nao outro?

**AINDA NAO PREPARADO** — precisa estudar.

Voce testou 5 dias (Stage 5b) e foi pior. Precisa explicar por que sentimento de noticias funciona melhor em horizonte medio.

---

## 8. Quais sao as limitacoes do trabalho?

**AINDA NAO PREPARADO** — precisa estudar.

Pontos que a banca pode querer ouvir:
- Dataset pequeno (~4000 amostras)
- Apenas uma acao (ITUB4) — resultados podem nao generalizar
- Apenas uma fonte de noticias (InfoMoney)
- Nao testou se as previsoes seriam lucrativas na pratica (sem backtest)
- Desbalanceamento de classes (57/43)
