# Plano Refatoração e Evolução: TCC v2

Este documento detalha o plano de ação para elevar o projeto de conclusão de curso a um patamar de rigor científico publicável e a uma arquitetura de nível de produção. As ações visam corrigir vulnerabilidades metodológicas identificadas e modernizar a infraestrutura do sistema.

## Fase 1: Blindagem Metodológica (Rigor Científico)

O objetivo primário desta fase é eliminar qualquer viés de avaliação e garantir que os resultados do modelo reflitam o desempenho num ambiente de negociação realista.

### Passo 1.1: Implementar Expanding-Window Cross-Validation

Problema: O uso de uma janela estática (single-window) (ex: 70% treino / 15% validação / 15% teste) em séries temporais financeiras não-estacionárias gera resultados ilusórios e overfitting a um regime de mercado específico.

Ação: Substituir o particionamento estático por janelas deslizantes em expansão. O modelo treina nos meses $1$ a $t$ e testa no mês $t+1$. Na iteração seguinte, treina nos meses $1$ a $t+1$ e testa no mês $t+2$.

Impacto: Simula o fluxo contínuo do tempo real, garantindo que o modelo nunca tem acesso a dados futuros.

### Passo 1.2: Corrigir o Look-Ahead Bias (Vazamento de Dados Temporal)

Problema: O método atual aplica forward-fill cego. Se uma notícia é publicada às 19h (após o fecho do mercado), o seu sentimento é associado ao dia atual, vazando informação para a previsão do dia seguinte.

Ação: Alterar a granularidade do cruzamento (merge) de dados de diário para intradiário.

Regra de Negócio: Notícias publicadas após o horário de fecho da B3 (ex: 18h) só podem ser associadas às features de preço do pregão do dia seguinte.

### Passo 1.3: Estabelecer Baselines Verdadeiramente Ingénuas

Problema: A baseline atual (XGBoost com desfasamentos, volume e volatilidade) é, na verdade, um modelo autorregressivo competente.

Ação: Antes de avaliar qualquer rede neural, registar o desempenho de três modelos triviais:

Majoritário: Prevê sempre a classe dominante (ex: sempre "Alta").

Probabilístico: Prevê aleatoriamente, mas viciado com a probabilidade da classe dominante.

Inércia (Naïve): Prevê que a direção do preço de amanhã será idêntica à de hoje.

Impacto: Permite provar quantitativamente se os modelos avançados e os dados de sentimento superam efetivamente o acaso e a inércia do mercado.

## Fase 2: Experiências e Engenharia de Features

Esta fase foca-se em isolar variáveis para compreender o real impacto das features de texto.

### Passo 2.1: Teste de Ablação de Dimensionalidade

Problema: A transição do Ollama (1.024 dimensões) para o FinBERT (5 dimensões) alterou duas variáveis em simultâneo: a representação semântica e o número de dimensões. Não é claro se a melhoria adveio do conhecimento financeiro do modelo ou apenas da redução de ruído.

Ação: Comprimir os embeddings originais do modelo Ollama de 1.024 para exatamente 5 dimensões utilizando Análise de Componentes Principais (PCA).

Comparação: Treinar e comparar o modelo com os 5 componentes do Ollama versus as 5 features do FinBERT.

### Passo 2.2: Varredura de Horizontes (Horizon Sweep)

Problema: A alteração arbitrária da janela de previsão de 21 dias para 5 dias não fornece uma visão completa de quando o impacto das notícias é assimilado pelo mercado.

Ação: Implementar um ciclo de treino e teste iterando sobre múltiplos horizontes de previsão: $h \in \{1, 2, 5, 10, 21, 42\}$ dias.

Impacto: Identificar a janela temporal ideal onde o sentimento das notícias tem a máxima capacidade preditiva sobre o preço.

## Fase 3: Validação de Modelos e Expansão de Dados

Com a metodologia corrigida, o foco passa para a extração do máximo poder preditivo.

### Passo 3.1: Escrutínio da Rede Convolucional Temporal (TCN)

Problema: A arquitetura TCN demonstrou promessa (AUC de 0.643) em avaliações secundárias, mas não foi sujeita ao rigor analítico do modelo principal.

Ação: Submeter o modelo TCN ao protocolo completo: validação expanding-window, avaliação com múltiplas seeds e análise de variância estatística.

### Passo 3.2: Ingestão de Múltiplas Fontes de Notícias

Problema: Depender exclusivamente de um portal (InfoMoney) enviesa os dados para uma linha editorial específica, possivelmente capturando reações tardias focadas no retalho.

Ação: Desenvolver e integrar scrapers adicionais para fontes institucionais, como:

Comunicados oficiais da CVM.

Artigos do Valor Económico.

Fios da Reuters Brasil.

## Fase 4: Modernização da Arquitetura (Ambiente de Produção)

Transição dos scripts exploratórios em notebooks para uma arquitetura de software robusta e em tempo real.

### Passo 4.1: Reestruturação em Monorepo

Ação: Migrar a aplicação para uma estrutura de monorepo (ex: com Turborepo), dividindo o sistema em pacotes independentes:

apps/scraper: Serviço de cron jobs para a ingestão contínua de notícias.

apps/inference: API para extração de sentimento e predições do modelo.

apps/dashboard: Interface web (Next.js) para visualização.

Stack: TypeScript ponta a ponta.

Base de Dados: Utilizar o Prisma (ORM) com uma base de dados relacional (ex: PostgreSQL) para modelar rigorosamente os timestamps e a relação 1:N entre dias de negociação e notícias.

### Passo 4.2: Ingestão e Processamento em Tempo Real com AI SDK

Ação: Substituir a extração em lote por um fluxo orientado a eventos. A entrada de uma nova notícia aciona imediatamente o pipeline de processamento.

Inovação: Integrar o Vercel AI SDK para não só calcular o sentimento global (via FinBERT ou um LLM rápido), mas também para extrair entidades específicas (empresas concorrentes citadas, pessoas-chave) num formato JSON tipado (zod), enriquecendo as features disponíveis para o modelo de previsão.
