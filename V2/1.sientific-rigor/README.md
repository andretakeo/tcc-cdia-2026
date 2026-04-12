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
