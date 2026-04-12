# Conclusões Finais: O Valor do Rigor Científico e da Arquitetura em Machine Learning Financeiro

Após a refatorização completa do projeto para a sua versão final (V2) e a submissão dos modelos a um escrutínio científico rigoroso, este trabalho conclui-se com descobertas fundamentais tanto no domínio financeiro como na engenharia de dados.

## 1. A Ilusão do Sinal e a Eficiência do Mercado
A avaliação inicial (V1) sugeria que o sentimento de notícias financeiras do retalho poderia prever a direção do preço da ITUB4 com um ROC-AUC de 0.709. Contudo, a aplicação de protocolos de validação estritos na V2 (Expanding-Window Cross-Validation e a "Regra das 18h") revelou que este resultado era um artefacto gerado por fuga temporal de dados (Look-Ahead Bias) e overfitting a uma janela única.

Os resultados finais da validação com a rede Temporal Convolutional Network (TCN), utilizando 20 sementes aleatórias, ditaram o seguinte:
- **Baseline de Inércia (Momento do Preço):** AUC 0.6058
- **TCN com Dados Purificados (Média):** AUC 0.5621 (± 0.08)
- **Significância Estatística:** O modelo TCN não superou a inércia do mercado de forma consistente (p = 0.48).

**Conclusão Empírica:** A hipótese de que o sentimento de notícias extraídas de portais de retalho (como o InfoMoney) possui poder preditivo superior à inércia do preço para ativos de altíssima liquidez (como a ITUB4) é rejeitada. O mercado precifica estas informações públicas de forma demasiado eficiente para que um modelo diário consiga extrair alfa consistente apenas com dados textuais de consumo geral.

## 2. A Honestidade como Contributo Científico
Numa área frequentemente inundada por backtests sobre-otimizados e resultados ilusórios, o maior contributo analítico deste trabalho é a desconstrução do seu próprio viés. A transição de um falso positivo (0.709) para a identificação correta da realidade estatística (0.562) demonstra o perigo das métricas isoladas e a importância vital de estabelecer baselines ingénuas e probabilísticas antes de se adotarem redes neuronais complexas.

## 3. O Legado Arquitetural: Uma Plataforma Pronta para Alfa
Se a predição da ITUB4 via notícias de retalho falhou, o sistema construído para o provar é um triunfo de engenharia de software. O projeto evoluiu de notebooks de exploração estáticos para uma Plataforma Quantitativa de Grau de Produção:
- **Pipeline em Cascata:** A implementação de triagem por IA (Vercel AI SDK com gpt-4o-mini) provou ser essencial para filtrar o ruído macroeconómico antes da extração profunda de sentimento, garantindo uma densidade de sinal muito superior na base de dados.
- **Resiliência de Dados:** O esquema Prisma (PostgreSQL) automatiza a correção de viés de retrospetiva, associando deterministicamente os eventos após o fecho do mercado ao dia útil de negociação subsequente.
- **Monorepo Orientado a Eventos:** A orquestração entre Scraper, Inference Pipeline e Dashboard permite testes rápidos de novas hipóteses de mercado em tempo real.

## 4. Perspetivas de Trabalho Futuro
A infraestrutura está pronta e validada; o próximo passo lógico é substituir a fonte do sinal. As direções futuras mais promissoras incluem:
- **Dados Institucionais e de Alta Frequência:** Plugar o pipeline de inferência a feeds da Bloomberg, Reuters ou comunicados da CVM com resolução ao minuto, tentando antecipar a reação institucional.
- **Ativos de Menor Liquidez:** Aplicar a arquitetura a Small Caps da B3, onde a ineficiência do mercado é maior.
- **Análise de Entidades:** Focar em como o cruzamento das entidades extraídas afeta setores inteiros simultaneamente.

**A V2 provou que o método está correto. O caminho para o alfa agora reside apenas na busca pelos dados certos.**
