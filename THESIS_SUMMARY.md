# Resumo Executivo do TCC — Guia para Defesa

## O que você fez

Você construiu um pipeline completo que coleta notícias financeiras brasileiras (InfoMoney), extrai sentimento com um modelo de IA especializado (FinBERT-PT-BR), e tenta prever se ações da B3 vão subir ou descer. Testou em 3 ações: ITUB4, PETR4 e VALE3.

## A história em 4 atos

### Ato 1 — "Embeddings genéricos não funcionam" (Etapas 1-3)

Você coletou ~5.900 artigos do InfoMoney e transformou cada um em um vetor de 1.024 números usando um modelo genérico (Ollama). Combinou com dados de preço do Yahoo Finance e treinou 4 modelos (BiLSTM, Transformer, XGBoost). **Nenhum funcionou** — o melhor deu AUC 0,610, pouco acima do acaso. Conclusão: o modelo de linguagem genérico captura "significado geral" do texto, mas a maioria é irrelevante para prever preços.

### Ato 2 — "Sentimento específico parece funcionar!" (Etapa 4)

Você trocou os 1.024 embeddings genéricos por apenas 5 features de sentimento financeiro do FinBERT-PT-BR (positivo/negativo/neutro). O Transformer saltou para **AUC 0,709** — aparentemente um resultado muito bom. Parecia que o sentimento financeiro específico era a chave.

### Ato 3 — "Será que funciona mesmo?" (Etapa 5 / Capítulo 5)

Você desconfiou do resultado e fez algo raro em trabalhos de graduação: **testou se o próprio resultado era confiável**. Em 1.435+ execuções de modelo, descobriu que:

- O AUC 0,709 **não é reproduzível** — rodando o mesmo código com outra semente, dá 0,442
- O Transformer é **instável**: com 20 sementes diferentes, o AUC varia de 0,08 a 0,93 (!!!)
- Um modelo simples com 5 features de preço (sem notícia nenhuma) dá **AUC 0,658** — quase igual
- Sob avaliação rigorosa (expanding-window CV), o sentimento adiciona **+0,003** — estatisticamente zero (p = 0,49)
- O TCN (melhor resultado prático, AUC 0,643) também cai para 0,556 sob avaliação rigorosa

### Ato 4 — "O resultado real é outro" (Conclusão)

O sentimento FinBERT-PT-BR do InfoMoney **não ajuda** a prever preços nestas condições. O AUC 0,709 era uma **ilusão** causada por avaliar em uma única janela temporal. A contribuição real do trabalho é **mostrar como a avaliação por janela única engana** — e propor protocolos para evitar isso.

---

## O que você defende

### Contribuição 1: O pipeline funciona (mesmo que o resultado seja nulo)

Você construiu um sistema completo e reproduzível: coleta → sentimento → features → modelos. O código está versionado, os dados estão disponíveis, qualquer pessoa pode replicar. Isso tem valor técnico independente do resultado preditivo.

### Contribuição 2: A autocorreção metodológica

A maioria dos trabalhos pararia no AUC 0,709 e declararia sucesso. Você foi além e mostrou que o resultado é um artefato. Isso é **raro e valorizado** pela banca — demonstra maturidade científica.

### Contribuição 3: Demonstração empírica de um problema conhecido

A literatura já sabia que avaliação por janela única é problemática (Bailey 2014, López de Prado 2018). Mas poucos trabalhos **demonstram** isso com dados reais. Você fez com 1.435 execuções, 3 ativos, 6 protocolos diferentes, e testes estatísticos formais.

### Contribuição 4: Protocolos mínimos propostos

Você propõe 6 regras para pesquisa em ML financeiro:
1. Sempre reportar intervalos de confiança bootstrap
2. Treinar com ≥10 sementes
3. Usar expanding-window CV (não split único)
4. Comparar contra baseline autoregressivo
5. Monitorar distribuição de predições (não só AUC)
6. Auditar correlação validação-teste

---

## Perguntas prováveis da banca e como responder

### "Por que não evitou o protocolo errado desde o início?"

> "O walk-forward split único é o padrão na literatura — Fischer & Krauss 2018, Xu & Cohen 2018, Araci 2019 todos usam. Adotei por comparabilidade. O Capítulo 5 demonstra empiricamente por que esse padrão é problemático."

### "O baseline não é trivial — por que chamar de baseline?"

> "Correto, renomeei para 'baseline autoregressivo'. Também avaliei baselines verdadeiramente ingênuos (classe majoritária AUC=0,500, coin flip AUC=0,500, persistência AUC=0,474) para mostrar a hierarquia completa."

### "E se a representação do sentimento fosse diferente?"

> "Minha conclusão é restrita: estas 5 features do FinBERT-PT-BR, com esta fonte (InfoMoney), nestes 3 ativos, nestes horizontes (5-21 dias). Outras representações (aspect-based, sentence-level), outras fontes (CVM, Bloomberg), outros ativos podem dar resultados diferentes. Isso está nas direções futuras."

### "O forward-fill não causa look-ahead bias?"

> "Discuto isso na Seção 4.3. Artigos pós-mercado são atribuídos ao dia t em vez de t+1. Três fatores atenuam: horizonte de 5-21 dias dilui 1 dia de deslocamento, média diária dilui artigos individuais, e o forward-fill propaga na direção causal correta. Em horizontes intraday seria necessário filtro explícito por hora."

### "E o confundimento dimensionalidade vs domínio?"

> "Controlei com um experimento: 20 subconjuntos aleatórios de 5 dimensões dos embeddings Ollama deram AUC médio de 0,509 — perto do acaso. O FinBERT com 5 dimensões deu 0,670. A melhoria é de domínio, não de dimensionalidade. Mas sob avaliação multi-fold, essa vantagem desaparece."

### "O TCN não foi validado com o mesmo rigor?"

> "Foi — adicionei no Experimento 5.10b. Multi-seed: 0,513 ± 0,102 (60% das sementes abaixo de 0,50). Expanding-window CV: 0,556, abaixo do baseline. O AUC 0,643 original era outro artefato de janela única."

---

## Números-chave para ter na cabeça

| Métrica | Valor |
|---|---|
| Artigos coletados | 5.872 |
| Execuções de modelo | 1.435+ |
| AUC "headline" (janela única) | 0,709 |
| AUC real (expanding-window CV) | ~0,51 |
| Ganho do sentimento (ablation) | +0,003 (p=0,49) |
| Baseline autoregressivo | 0,658 |
| Std do Transformer (20 seeds) | 0,261 |
| Std do baseline (20 seeds) | 0,012 |
| Ativos testados | 3 (ITUB4, PETR4, VALE3) |
| Folds no deep-dive VALE3 | 52 folds × 10 seeds = 880 runs |
