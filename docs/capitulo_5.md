# Capítulo 5 — Investigação metodológica e o viés da avaliação por janela única

## 5.1 Motivação

O que significa, de fato, afirmar que um modelo de *machine learning* "funciona" para predição de direção de preços? Uma métrica pontual como ROC-AUC = 0.709, reportada sobre um único conjunto de teste, responde a essa pergunta apenas parcialmente: ela descreve o desempenho do modelo em *uma amostra particular*, sob *uma semente particular*, sob *uma única divisão temporal* dos dados. Em séries financeiras, onde a não-estacionariedade é a regra e o volume de dados é tipicamente pequeno em relação à capacidade dos modelos modernos, esta descrição pode ser enganosa em grau suficiente para inverter a conclusão qualitativa do estudo.

Este capítulo é um estudo de caso dessa inversão. Ele parte do resultado apresentado nos Capítulos 3 e 4 — um Transformer treinado com *features* de sentimento FinBERT-PT-BR atingindo ROC-AUC = 0.709 e acurácia de 76.3% na previsão de direção do ITUB4 em horizonte de 21 dias — e aplica um conjunto progressivamente mais rigoroso de protocolos de avaliação para determinar se o resultado sobrevive a escrutínio metodológico. O protocolo original utilizado nos capítulos anteriores é o padrão da literatura: *walk-forward split* único com divisão cronológica 70% treino / 15% validação / 15% teste.

Três observações sobre o resultado original justificam a investigação:

1. A matriz de confusão do Transformer-FinBERT indica `precision(Desce)=1.00` mas `recall(Desce)=0.20` — o modelo prevê "Desce" apenas 11 vezes em 177 amostras de teste. Em 166 das 177 amostras, prevê "Sobe". Esta é uma distribuição de predições altamente degenerada, próxima de um classificador que sempre prevê a classe majoritária.

2. A acurácia de 76.3% é muito próxima da proporção de dias de alta no conjunto de teste (~69%). A diferença líquida sobre um preditor constante que sempre prevê "Sobe" é de aproximadamente 7 pontos percentuais.

3. Não foram reportados intervalos de confiança, nem resultados de múltiplas sementes aleatórias, nem comparação contra um *baseline* autoregressivo trivial. A ausência destes controles é comum na literatura mas deixa o resultado aberto a questionamento.

A sequência de oito experimentos documentada aqui responde à pergunta **o resultado original é reproduzível e confiável sob avaliação rigorosa?** A resposta, antecipada aqui para transparência do leitor, é negativa: sob validação cruzada *expanding-window* com múltiplas sementes e intervalos de confiança, o Transformer-FinBERT não supera um *baseline* XGBoost de cinco *features* autoregressivas de preço em nenhum dos três ativos testados (ITUB4, PETR4, VALE3), e o AUC = 0.709 originalmente reportado é mostrado como um artefato de amostragem de janela única combinado com colapso bimodal da arquitetura. Esta inversão do diagnóstico original é o achado metodológico central da dissertação e motiva o reposicionamento da contribuição discutido na Seção 5.12.

## 5.2 Protocolo geral dos experimentos

Todos os experimentos deste capítulo compartilham o mesmo conjunto de dados base (features de preço do Yahoo Finance + features de sentimento FinBERT-PT-BR, conforme descrito nos Capítulos 2 e 4), o mesmo alvo binário (1 se `Close[t+21] > Close[t]`, caso contrário 0), e a mesma arquitetura Transformer do Capítulo 4 (encoder com 2 camadas, 4 cabeças de atenção, `d_model=64`, dropout 0.2, positional encoding sinusoidal, mean pooling temporal, janelas de 30 dias).

O código e os resultados numéricos completos de cada experimento estão em `9.baselines/`:

| Experimento | Notebook | Runs |
|---|---|---:|
| 5.3 — Dumb baseline | `dumb_baseline.ipynb` | 4 |
| 5.4 — Bootstrap CI | `bootstrap_stage4.ipynb` | 1 |
| 5.5 — Multi-seed em ITUB4 | `multi_seed.ipynb` | 40 |
| 5.6 — Multi-seed × multi-ticker | `multi_seed_multi_ticker.ipynb` | 120 |
| 5.7 — Ensemble + backtest | `ensemble_backtest.ipynb` | 20 |
| 5.8 — Expanding-window CV | `expanding_window_cv.ipynb` | 145 |
| 5.9 — VALE3 deep-dive | `vale3_deepdive.ipynb` | 880 |
| 5.10 — Ablation PRICE / SENT / PRICE+SENT | `ablation_price_vs_sentiment.ipynb` | 225 |

Total: **1.435 treinamentos de modelo**, cobrindo 3 ativos, até 20 sementes por experimento, e até 52 janelas de teste por ativo.

Os utilitários reusáveis (split walk-forward, bootstrap CI, geração de alvo binário) estão em `9.baselines/eval_utils.py`.

## 5.3 Experimento 1 — Dumb baseline: qual o piso sem sentimento?

**Pergunta.** Se o Transformer-FinBERT atinge AUC = 0.709, qual é o AUC de um modelo trivial que não usa features textuais nenhuma?

**Protocolo.** Treinar 4 modelos (Logistic Regression, XGBoost, BiLSTM reduzido, Transformer pequeno) usando apenas **5 features autoregressivas** extraídas da série de preços: `return`, `lag_1`, `lag_5`, `Volume`, `std21`. Mesmo *split* walk-forward 70/15/15 do experimento original, mesmo alvo binário. Toda métrica reportada com **intervalo de confiança bootstrap 95%** sobre 1.000 reamostragens do conjunto de teste.

**Resultado.** O XGBoost atingiu **AUC = 0.658 [0.565, 0.744]**, acompanhado por uma matriz de confusão também degenerada (*precision(Desce)*=0.31, *n_down_calls*=177/178). O gap contra o resultado original da Etapa 4 é de apenas **0.051 pontos de AUC**, e o intervalo de confiança do *baseline* se aproxima perigosamente de 0.709.

**Interpretação.** Um modelo clássico treinado em 5 features de preço, sem qualquer informação textual, captura 93% do AUC reportado pelo modelo com sentimento. Este já é um sinal de alerta: se o ganho incremental da representação textual é de apenas 5 pontos de AUC e nenhum intervalo de confiança foi reportado, não há evidência de que o ganho seja estatisticamente distinguível do ruído de estimação.

## 5.4 Experimento 2 — Reprodução com intervalo de confiança

**Pergunta.** O próprio AUC = 0.709 da Etapa 4, quando reportado com intervalo de confiança bootstrap, é consistente ou não com o *baseline*?

**Protocolo.** Reconstruir o Transformer-FinBERT original (mesma arquitetura, mesmas features, mesmo *split*, semente fixa em 42) e aplicar o mesmo `bootstrap_auc_ci` usado no experimento anterior.

**Resultado.** Sob recompilação com semente 42, o modelo atingiu **AUC = 0.442 [0.358, 0.528]**. A matriz de confusão mostra `[[36, 0], [116, 0]]` — o modelo prevê "Sobe" para todas as 152 amostras do conjunto de teste.

**Interpretação.** Com apenas uma diferença de semente aleatória em relação ao experimento original, o mesmo código e a mesma arquitetura produzem um AUC de 0.442 em vez de 0.709 — uma diferença de **0.267 pontos de AUC**. Esta é uma variação absurda para uma mudança de semente. O experimento original não é reproduzível no sentido estrito: não há um AUC "verdadeiro" que diferentes execuções convergem para, mas sim uma distribuição ampla da qual o valor 0.709 é apenas uma amostra.

## 5.5 Experimento 3 — Multi-seed em ITUB4

**Pergunta.** Qual é a distribuição completa de AUCs que o Transformer-FinBERT produz quando treinado com sementes diferentes?

**Protocolo.** Fixar todos os outros aspectos (dados, *split*, arquitetura, otimizador) e variar apenas a semente de inicialização. Treinar 20 vezes o Transformer-FinBERT e 20 vezes o *baseline* XGBoost sobre o mesmo conjunto de treino/validação/teste do ITUB4. Para cada execução, registrar: AUC, *precision*/*recall* por classe, número de predições "Desce" emitidas.

**Resultado.** Tabela de estatísticas agregadas sobre 20 sementes:

| Modelo | AUC mean | AUC std | AUC median | AUC min | AUC max | prec(Desce) mean |
|---|---:|---:|---:|---:|---:|---:|
| **Transformer + FinBERT** | 0.686 | **0.261** | **0.802** | 0.080 | 0.931 | 0.177 |
| Dumb baseline (XGB) | 0.641 | **0.012** | 0.638 | 0.615 | 0.660 | 0.313 |

Dois fatos merecem destaque:

- **A mediana do Transformer (0.802) é superior ao resultado original (0.709).** Longe de ter sido um valor ótimo, 0.709 está **abaixo** da mediana da distribuição. 13 de 20 sementes (65%) atingem AUC ≥ 0.709.
- **O desvio-padrão do Transformer é 22× maior que o do baseline.** O *baseline* produz AUCs entre 0.615 e 0.660 — uma faixa estreita e estável. O Transformer produz AUCs entre 0.080 e 0.931 — essencialmente toda a faixa possível.

O *scatter plot* de AUC contra número de *down calls* (ver `multi_seed_tradeoff.png`) revela a estrutura responsável pela variância: **os pontos de alta AUC do Transformer agrupam-se em dois extremos**, um com zero *down calls* (sempre prevê "Sobe") e outro com ~150 *down calls* (sempre prevê "Desce"). Não há um "meio-termo" discriminativo. O modelo está oscilando entre dois estados degenerados.

**Interpretação.** O AUC = 0.709 não é nem um valor de sorte, nem um valor típico — é um ponto aleatório amostrado de uma distribuição extremamente ampla e bimodal. A reprodutibilidade efetiva é zero: cada execução converge a um mínimo local diferente, e o AUC resultante depende da sorte de inicialização mais do que de qualquer propriedade aprendida das *features*.

## 5.6 Experimento 4 — Multi-seed em múltiplos ativos

**Pergunta.** O padrão observado em ITUB4 se replica em PETR4 e VALE3? Ou o AUC elevado é específico ao setor bancário?

**Protocolo.** Repetir o experimento 5.5 sobre 3 ativos, 20 sementes cada, ambos os modelos. Total: 120 treinamentos. Dados de preço obtidos via `yfinance` (período de 5 anos), *sentimento* obtido dos arquivos já computados na Etapa 4 (`{itub4,petr4,vale3}_daily_sentiment.csv`). *Split* e arquitetura idênticos.

**Resultado.** Tabela de medianas de AUC por ativo e modelo:

| Ativo | Δ balance (test−train) | Baseline XGB mediana | Transformer mediana | Δ mediana |
|---|---:|---:|---:|---:|
| **ITUB4** | +0.117 | 0.682 | **0.801** | **+0.119** |
| **PETR4** | −0.030 | 0.587 | **0.334** | **−0.253** |
| **VALE3** | +0.342 | 0.679 | **0.992** | **+0.313** |

**Interpretação.** O achado de ITUB4 não é específico ao setor bancário — é específico à combinação de arquitetura (Transformer com 16 features em ~800 amostras de treino) e protocolo de avaliação (janela única). O *baseline* tem comportamento estável e razoavelmente discriminativo nos três ativos (~0.59–0.68), enquanto o Transformer oscila entre 0.33 e 0.99 dependendo exclusivamente do ativo. Esta diferença de 0.66 pontos de AUC para o mesmo modelo, treinado com o mesmo código, sobre a mesma arquitetura, é grande demais para ser atribuída a propriedades intrínsecas dos ativos. A origem do fenômeno é investigada em detalhe pelo experimento 5.8 (avaliação *multi-fold*) e pelo experimento 5.9 (*deep-dive* em VALE3).

## 5.7 Experimento 5 — Ensemble e backtest: o sinal se traduz em valor prático?

**Pergunta.** Se a variância do Transformer vem de *seed noise*, um *ensemble* de múltiplas sementes suavizaria o colapso bimodal? E o sinal resultante seria utilizável como estratégia *long/flat*?

**Protocolo.** Treinar 20 sementes do Transformer-FinBERT sobre ITUB4, armazenando as probabilidades previstas por dia (não apenas AUC). Filtrar sementes que falham na validação (val AUC ≥ 0.5), usando apenas o conjunto de validação — nunca o teste — para evitar seleção tendenciosa. Calcular a média das probabilidades das sementes sobreviventes e otimizar o limiar de decisão no conjunto de validação via *macro F1*. Finalmente, rodar uma estratégia *long/flat*: *long* ITUB4 exceto quando o *ensemble* emite sinal de "Desce", então caixa por 21 dias. Custo de transação: 10 *basis points* por troca de posição.

**Resultado.** Apenas **2 de 20 sementes** passaram no filtro de validação (val AUC ≥ 0.5). Quando a decisão é feita honestamente com base na validação, as sementes filtradas são justamente as que **falham no teste**: AUC médio do *ensemble* = 0.137. No *backtest*:

| Estratégia | Retorno total | Retorno anualizado | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Buy-and-hold ITUB4 | **+50.30%** | **+96.51%** | **3.25** | −4.90% |
| *Long/flat* ensemble | −0.10% | −0.17% | −1.29 | 0.00% |

O *ensemble* permanece fora do mercado por 100% do período de teste. *Buy-and-hold* vence decisivamente em todas as métricas.

**Interpretação.** Este resultado contém uma lição metodológica crucial: **a correlação entre AUC de validação e AUC de teste é negativa**. Das 20 sementes treinadas, 18 têm val AUC < 0.5 mas test AUC entre 0.70 e 0.93. As duas sementes que "parecem boas na validação" são exatamente aquelas que colapsaram no estado degenerado oposto ao exigido pelo teste. Se um pesquisador selecionar o modelo pela validação (prática padrão), escolherá o pior modelo para o teste. Esta anticorrelação é assinatura de regime non-stationary severo entre as janelas de validação e teste, e é invisível sob avaliação de janela única.

O que parecia ser um detector de cauda de alta precisão (*precision(Desce)* = 1.00 no experimento original) não se traduz em estratégia *long/flat* lucrativa, porque:

1. A alta precisão em *down calls* é uma coincidência entre a raridade do evento previsto e a raridade da predição — não é sinal real.
2. Quando o modelo é forçado a ser consistente via *ensemble* e seleção honesta de sementes, o sinal desaparece.

## 5.8 Experimento 6 — Expanding-window cross-validation: a avaliação correta

**Pergunta.** Qual é o desempenho dos dois modelos sob um protocolo de avaliação que não depende de uma única janela de teste arbitrária?

**Protocolo.** Substituir o *split* único 70/15/15 por **validação cruzada *expanding-window***, adequada para séries temporais não-estacionárias:

- Janela mínima de treino: 600 dias
- Janela de validação: 90 dias consecutivos
- Janela de teste: 90 dias consecutivos após a validação
- Passo entre *folds*: 90 dias
- Repetições: 5 *folds* por ativo × 5 sementes por *fold* = 25 pontos por (ativo, modelo)

Total: 3 ativos × 5 *folds* × 5 sementes × 2 modelos = **145 treinamentos** (alguns *folds* são descartados quando o conjunto de teste tem classe única). Dados e *features* idênticos aos experimentos anteriores.

**Resultado.** Médias de AUC agregadas sobre todos os *folds* e sementes:

| Ativo | Baseline XGB | Transformer + FinBERT | Δ (trans − base) |
|---|---:|---:|---:|
| **ITUB4** | **0.700** | 0.445 | **−0.255** |
| **PETR4** | **0.702** | 0.447 | **−0.255** |
| **VALE3** | 0.599 | **0.635** | +0.036 |
| **Média** | **0.667** | **0.509** | **−0.158** |

A inversão é dramática:

1. **O dumb baseline vence o Transformer-FinBERT em 2 dos 3 ativos por uma margem de 0.25 pontos de AUC.** Isto é uma diferença enorme, muito maior do que qualquer ganho do Transformer sob avaliação de janela única.
2. **O Transformer opera próximo do acaso (AUC ~ 0.5) em média.** Em ITUB4 e PETR4, atinge AUCs menores que 0.5 na média dos *folds*.
3. **Em VALE3, o Transformer aparentemente vence por 0.036 pontos de AUC**, mas este resultado é investigado em profundidade no experimento 5.9 e revelado como ruído amostral.

O teste da hipótese de *class-prior shift* (correlação entre AUC e desvio do balanço train→test) produz valores de Pearson entre −0.46 e +0.15 em todos os (ticker, modelo), todos com p > 0.4 sobre 5 pontos. A hipótese é rejeitada: o *shift* não é o fator explicativo principal.

**Interpretação.** Sob avaliação que amostra múltiplas janelas ao longo do tempo, o efeito de sorte amostral da janela única é eliminado. O que resta é o desempenho real dos modelos, e esse desempenho real é: o *baseline* é um classificador fraco-mas-honesto (AUC ~ 0.67), e o Transformer-FinBERT é um classificador estruturalmente degenerado (AUC ~ 0.51, mas com distribuição bimodal conforme se verá a seguir).

A figura `expanding_cv_overtime.png` torna o resultado visual: em todos os 3 ativos, a linha vermelha (baseline) fica consistentemente acima da linha azul (Transformer) ao longo de quase todos os *folds*, com barras de erro sistematicamente menores no *baseline*. Esta é a **figura hero** deste capítulo.

## 5.9 Experimento 7 — VALE3 deep-dive: investigando a única exceção aparente

**Pergunta.** VALE3 foi o único ativo onde o Transformer pareceu marginalmente competitivo no experimento 5.8 (+0.036 de AUC). Este ganho é real, ou é ruído amostral de apenas 5 *folds*?

**Protocolo.** Executar o mesmo protocolo *expanding-window* exclusivamente em VALE3, mas com parâmetros muito mais finos: janela de validação 60 dias, teste 60 dias, passo 60 dias, histórico máximo (`period=max`), 10 sementes por *fold*. Total: **52 *folds* × 10 sementes × 2 modelos = 880 treinamentos** (os *folds* sem as duas classes no teste são descartados, restando 39 *folds* pareados). Aplicar teste não-paramétrico de Wilcoxon *signed-rank* pareado por *fold* entre os AUCs dos dois modelos.

**Resultado.** Estatísticas pareadas sobre 39 *folds*:

| Estatística | Valor |
|---|---:|
| Transformer AUC mean (sobre *folds*) | 0.484 |
| Baseline AUC mean (sobre *folds*) | 0.535 |
| Δ médio (trans − base) | **−0.051** |
| Folds onde trans > base | **14 / 39 (36%)** |
| Wilcoxon *signed-rank* p-value | **0.194** |
| Bootstrap 95% CI sobre Δ médio | **[−0.136, +0.033]** |
| CI contém zero? | **Sim** |

Com uma amostra 8× maior que no experimento 5.8, o sinal de VALE3 **inverte**: o *baseline* passa a vencer o Transformer por 0.051 pontos de AUC, e o Transformer vence apenas em 36% dos *folds* — menos da metade. A diferença não é estatisticamente significativa (p = 0.194), e o intervalo de confiança *bootstrap* contém zero.

**A figura `vale3_deepdive_hist.png` contém o achado mais importante de todo este capítulo.** Ela plota a distribuição completa de 880 AUCs (52 *folds* × 10 sementes, para cada modelo):

- **O dumb baseline produz uma distribuição unimodal**, centrada próxima de 0.55, com massa entre 0.1 e 0.9. É a forma que se esperaria de um classificador fraco-mas-honesto enfrentando um problema difícil.
- **O Transformer produz uma distribuição bimodal severa**, com um pico de ~55 execuções em AUC < 0.05 (o maior *bin* do histograma) e um pico menor de ~37 execuções em AUC > 0.95. A massa central em AUC ≈ 0.5 é reduzida. A média de 0.484 é estatisticamente correta mas descritivamente enganosa: quase nenhuma execução produz de fato um AUC próximo de 0.48.

**Interpretação.** O Transformer-FinBERT não está aprendendo uma função preditiva. Está aprendendo um classificador degenerado (*sempre prever Desce* ou *sempre prever Sobe*), cuja escolha entre os dois estados depende do ruído de inicialização. Quando a escolha coincide com a classe majoritária do *fold* de teste, o AUC aproxima-se de 1.0. Quando não coincide, o AUC aproxima-se de 0.0. A média populacional sobre muitos *folds* e sementes se cristaliza em ~0.5, que é a assinatura esperada de um preditor sem capacidade discriminativa real.

A existência de centenas de execuções com AUC ≥ 0.9 em VALE3 é o que permite que avaliações de janela única ocasionalmente reportem AUCs altos — mas estas são, estatisticamente, coincidências estruturais, não capacidade preditiva.

## 5.10 Experimento 8 — Ablation: o sentimento adiciona algo, sob protocolo correto?

**Pergunta.** Os experimentos 5.3 a 5.9 estabeleceram que o Transformer-FinBERT (16 *features* combinadas) não supera o *baseline* XGBoost (5 *features* de preço). Mas resta uma pergunta separada e mais focada: **se isolarmos a contribuição incremental do sentimento, sob o mesmo modelo (XGBoost) e o mesmo protocolo (CV *expanding-window*), o sentimento adiciona algum sinal mensurável?** Esta é a versão metodologicamente correta da comparação Etapa 3 vs Etapa 4 dos capítulos anteriores.

**Protocolo.** Treinar o XGBoost (mesma configuração de 5.3 e 5.8) em três configurações de *features*, cada uma sob *expanding-window CV* com 5 *folds* e 5 *seeds* nos 3 ativos:

| Configuração | Features | Dim |
|---|---|---:|
| **PRICE** | `return`, `lag_1`, `lag_5`, `Volume`, `std21` | 5 |
| **SENT** | `n_articles`, `mean_logit_pos`, `mean_logit_neg`, `mean_logit_neu`, `mean_sentiment` | 5 |
| **PRICE+SENT** | união das duas | 10 |

Total: 3 ativos × 5 *folds* × 5 sementes × 3 configurações = **225 treinamentos**. A comparação chave é PRICE+SENT vs PRICE, com teste de Wilcoxon *signed-rank* pareado e intervalo de confiança *bootstrap* sobre o delta médio.

**Resultado.** Médias de AUC sobre todos os *folds* e sementes:

| Ativo | PRICE | SENT | PRICE+SENT | Δ (P+S − P) |
|---|---:|---:|---:|---:|
| **ITUB4** | 0.684 | 0.436 | 0.651 | **−0.033** |
| **PETR4** | 0.692 | 0.494 | 0.676 | **−0.016** |
| **VALE3** | 0.609 | 0.510 | 0.667 | **+0.058** |
| **Média global** | **0.662** | **0.480** | **0.665** | **+0.003** |

Estatística pareada PRICE+SENT vs PRICE sobre 75 pares (3 ativos × 5 *folds* × 5 sementes):

- Δ médio = **+0.003**
- Pares onde PRICE+SENT > PRICE: **32 / 75 (43%)**
- Wilcoxon *signed-rank* p-value = **0.4941**
- *Bootstrap* 95% CI sobre Δ médio: **[−0.012, +0.018]**
- CI contém zero? **Sim**

**Interpretação.** Os três achados deste experimento são, individualmente e em conjunto, conclusivos:

1. **O sentimento sozinho (SENT) opera abaixo do acaso** em todos os três ativos (média 0.480), o que indica que as cinco *features* derivadas do FinBERT-PT-BR — média dos *logits* positivo/negativo/neutro, classe média e contagem de artigos — não contêm sinal direcional preditivo no horizonte de 21 dias úteis. A média ligeiramente abaixo de 0.5 é compatível com ruído estatístico em torno do acaso.

2. **A combinação PRICE+SENT é estatisticamente indistinguível de PRICE sozinho.** O ganho médio de 0.003 pontos de AUC está bem dentro do intervalo de confiança *bootstrap*, é não-significativo no teste de Wilcoxon (p ≈ 0.49), e o sentimento "vence" em apenas 43% dos pares — pior que coin-flip.

3. **A direção do efeito é heterogênea entre ativos**: PRICE+SENT é *pior* que PRICE em ITUB4 (−0.033) e PETR4 (−0.016), e melhor em VALE3 (+0.058). A média se cancela. Esta heterogeneidade *poderia* ser interpretada como "o sentimento ajuda em mineração mas atrapalha em bancos", mas dado o tamanho amostral (25 *runs* por célula) e a magnitude dos efeitos (todos ≤ 0.06), a interpretação correta é ruído amostral.

A conclusão fecha definitivamente a investigação metodológica deste capítulo: **as features de sentimento FinBERT-PT-BR utilizadas neste estudo não adicionam sinal preditivo mensurável a um *baseline* autoregressivo simples, sob avaliação metodologicamente correta**. A aparente superioridade observada nos Capítulos 3 e 4 (Δ = 0.099 entre Etapas 3 e 4 com o Transformer) é totalmente explicada por (a) variância de avaliação por janela única e (b) propriedades de colapso bimodal do Transformer, não por contribuição informacional do sentimento.

Esta conclusão deve ser interpretada com cuidado: ela **não** afirma que sentimento de notícias financeiras seja, em geral, irrelevante para previsão de preços. Afirma apenas que (i) a representação específica adotada (5 *features* agregadas diariamente, derivadas do FinBERT-PT-BR), (ii) o horizonte específico (21 dias úteis) e (iii) os ativos específicos (3 *large caps* brasileiros) não exibem ganho preditivo mensurável quando comparados rigorosamente contra um *baseline* trivial. Outras representações, horizontes ou ativos podem produzir resultados diferentes — esta é uma das direções de trabalho futuro discutidas em 5.13.

## 5.11 Síntese dos achados

Os oito experimentos deste capítulo estabelecem uma sequência lógica de descobertas:

1. **(5.3–5.4)** O AUC = 0.709 original é acompanhado por uma matriz de confusão degenerada, não foi reportado com intervalo de confiança, e não é reproduzível sob mudança de semente.

2. **(5.5)** Sob 20 sementes diferentes em ITUB4, o mesmo modelo produz AUCs entre 0.08 e 0.93. A variância é explicada por colapso bimodal da arquitetura em dois estados degenerados.

3. **(5.6)** Os AUCs variam drasticamente entre ativos (PETR4 = 0.33, VALE3 = 0.99) sob avaliação de janela única.

4. **(5.7)** Um *ensemble* das sementes selecionadas por validação honesta gera uma estratégia *long/flat* que perde decisivamente para *buy-and-hold* (Sharpe −1.29 vs 3.25). Além disso, a correlação entre AUC de validação e AUC de teste é **negativa**.

5. **(5.8)** Sob validação cruzada *expanding-window* multi-fold, o dumb baseline vence o Transformer-FinBERT em 2 dos 3 ativos por 0.25 pontos de AUC, e a média global sobre todos os ativos é 0.667 (baseline) vs 0.509 (Transformer).

6. **(5.9)** O único caso aparentemente remanescente (VALE3) desaparece sob amostragem mais fina: com 880 execuções, a diferença não é estatisticamente significativa (Wilcoxon p = 0.194), e o Transformer apresenta a distribuição bimodal característica de classificador degenerado.

7. **(5.10)** Quando isolado em uma *ablation* formal sob CV multi-fold (XGBoost com PRICE / SENT / PRICE+SENT), as *features* de sentimento adicionam Δ médio = +0.003 ao *baseline* de preço (Wilcoxon p = 0.49, *bootstrap* CI [−0.012, +0.018] contém zero). Sentimento sozinho opera abaixo do acaso (AUC médio 0.480).

A conclusão unificada é inescapável:

> **Sob avaliação metodologicamente correta (multi-fold, multi-sementes, com intervalos de confiança e *ablation* de *features*), as *features* de sentimento FinBERT-PT-BR utilizadas neste estudo não adicionam sinal preditivo mensurável a um *baseline* autoregressivo de cinco *features* de preço, em nenhum dos três ativos brasileiros de grande capitalização testados. O resultado original de AUC = 0.709 em ITUB4 é um artefato da avaliação por janela única — uma combinação de variância de semente, viés de amostragem da janela de teste e colapso bimodal da arquitetura.**

## 5.12 Implicações metodológicas para pesquisa em ML financeiro

Os achados deste capítulo generalizam além do caso específico FinBERT-PT-BR / ITUB4. Eles sugerem que muitos resultados publicados na literatura de *deep learning* aplicado à predição de direção de preços podem sofrer dos mesmos problemas, e que a comunidade deveria adotar os seguintes protocolos mínimos:

1. **Reportar sempre intervalos de confiança bootstrap (95%) sobre ROC-AUC.** O tamanho típico de um conjunto de teste em séries financeiras (100–500 dias) gera desvios-padrão de AUC da ordem de 0.04–0.08. Diferenças menores que este valor não são evidência de nada.

2. **Treinar sempre com múltiplas sementes (≥ 10) e reportar média ± desvio-padrão.** Arquiteturas profundas em regimes de baixo volume de dados exibem colapso bimodal ou multi-modal; a semente importa mais do que qualquer hiperparâmetro.

3. **Usar *expanding-window cross-validation* em vez de *split* único.** O *split* único é estruturalmente enganoso em séries não-estacionárias, pois mede desempenho em uma única combinação arbitrária de regimes de treino e teste. O CV multi-fold amostra múltiplas combinações.

4. **Comparar sempre contra um *dumb baseline* autoregressivo.** Um XGBoost sobre 5 lags de preço é o piso correto. Se o modelo proposto não supera esse piso sob CV multi-fold, o ganho reportado é provavelmente ilusório.

5. **Monitorar a distribuição de predições e a matriz de confusão, não apenas a AUC.** Um AUC alto acompanhado de uma matriz de confusão degenerada (predição quase-constante) é um sinal de colapso para classe majoritária, não de aprendizagem real.

6. **Auditar a correlação entre AUC de validação e AUC de teste.** Correlações negativas ou nulas indicam *non-stationarity* severa entre as janelas, situação em que seleção de modelo tradicional é inviável.

## 5.13 Reposicionamento da contribuição da dissertação

À luz dos achados deste capítulo, propõe-se que a contribuição central da dissertação seja reposicionada: ao invés de apresentar um modelo de predição de direção de preços baseado em sentimento como resultado principal, a dissertação pode ser lida como uma investigação empírica sobre os limites da avaliação por janela única em *machine learning* aplicado a séries financeiras, usando o caso FinBERT-PT-BR / ITUB4 como estudo de caso. Nesta leitura, o *pipeline* construído nos Capítulos 1 a 4 permanece íntegro como artefato técnico — coleta de notícias, *embeddings*, extração de sentimento, engenharia de *features* e treinamento inicial — mas a interpretação dos resultados que ele produz é substancialmente revisada pelos achados do Capítulo 5.

Os argumentos a favor deste reposicionamento incluem:

- A evidência experimental é baseada em 1.435 execuções de modelos, cobrindo 3 ativos, sob seis protocolos de avaliação distintos (janela única, bootstrap CI, multi-seed, *expanding-window CV*, *deep-dive* multi-fold, *ablation* de *features*), com suporte estatístico formal (Wilcoxon *signed-rank*, intervalos de confiança *bootstrap*). A robustez empírica é alta.
- Os resultados são replicáveis: todos os *notebooks*, dados agregados e figuras estão disponíveis em `9.baselines/`, e qualquer pesquisador pode reproduzir a inversão a partir dos arquivos versionados.
- A magnitude da inversão é grande (0.709 sob janela única vs média 0.51 sob CV multi-fold), o que reduz o risco de que a diferença observada seja atribuível a variação amostral normal.
- Os protocolos de avaliação corrigidos (Seção 5.11) são diretamente aplicáveis por outros pesquisadores em trabalhos subsequentes na área, independentemente do contexto específico desta dissertação.

O reposicionamento não invalida o trabalho das etapas anteriores. O *pipeline* de coleta, processamento e treinamento é engenharia útil e replicável. O que é revisado é a leitura dos resultados numéricos que o *pipeline* produz: um achado que parecia ser sobre *representação textual para finanças* torna-se um achado sobre *viés metodológico em avaliação de séries temporais financeiras*. A decisão final sobre a adoção deste reposicionamento cabe ao(s) orientador(es) e à banca, e depende de considerações tanto científicas quanto de escopo do programa acadêmico.

## 5.14 Trabalhos futuros sugeridos

As conclusões deste capítulo abrem várias direções:

1. **Isolar o que, se algo, o sentimento adiciona.** Executar um *ablation* formal sob *expanding-window CV*: (a) apenas features de preço, (b) apenas features de sentimento, (c) preço + sentimento. A comparação direta entre (a) e (c) mede o ganho incremental *real* do sentimento após controle metodológico.

2. **Testar arquiteturas menos propensas a colapso.** A distribuição bimodal observada é característica de modelos com excesso de capacidade em relação ao sinal disponível. Modelos menores (Logistic Regression com interações, GBT leve) podem produzir classificadores honestos mesmo em regimes de baixo volume de dados.

3. **Generalizar o estudo para outros ativos e outros mercados.** O achado foi estabelecido para 3 ativos brasileiros. Replicá-lo em S&P 500 e mercados emergentes validaria a generalidade da conclusão metodológica.

4. **Estudar a anticorrelação validação-teste formalmente.** A descoberta de que seleção por validação escolhe o modelo errado para o teste é uma observação forte. Medir sua prevalência em diferentes janelas e mercados é uma direção de pesquisa autônoma e de alto impacto.

5. **Investigar fontes alternativas de sinal textual.** Se o sentimento agregado diário do InfoMoney não adiciona valor sob avaliação correta, outras fontes (Twitter, comunicados CVM, *analyst reports*) podem ter melhor razão sinal-ruído.

---

## Figuras referenciadas neste capítulo

| Figura | Arquivo | Seção | Conteúdo |
|---|---|---|---|
| Distribuição multi-seed em ITUB4 | `9.baselines/multi_seed_histograms.png` | 5.5 | Histogramas de AUC e *precision(Desce)* sobre 20 sementes |
| Trade-off AUC × *down calls* | `9.baselines/multi_seed_tradeoff.png` | 5.5 | *Scatter* revelando a bimodalidade |
| Distribuição por ticker | `9.baselines/multi_seed_multi_ticker.png` | 5.6 | 3 painéis de histograma, um por ativo |
| Backtest equity curves | `9.baselines/ensemble_backtest.png` | 5.7 | Curvas de capital *ensemble* vs *buy-and-hold* |
| Hero figure — CV over time | `9.baselines/expanding_cv_overtime.png` | 5.8 | AUC ± std por *fold*, 3 painéis |
| CV vs *class-prior shift* | `9.baselines/expanding_cv_hero.png` | 5.8 | *Scatter* AUC × shift, faceted por ativo |
| **VALE3 bimodal histogram** | `9.baselines/vale3_deepdive_hist.png` | 5.9 | **Distribuição de 880 execuções — figura central** |
| VALE3 por fold | `9.baselines/vale3_deepdive.png` | 5.9 | AUC por *fold* e delta pareado |
| Ablation boxplot | `9.baselines/ablation_boxplot.png` | 5.10 | PRICE vs SENT vs PRICE+SENT por ticker |

## Tabelas de resultados

Os CSVs brutos e agregados estão em `9.baselines/`:

- `results_dumb_baseline.csv` — Experimento 5.3
- `results_multi_seed.csv` + `multi_seed_summary.csv` — Experimento 5.5
- `results_multi_seed_multi_ticker.csv` + `multi_seed_multi_ticker_summary.csv` — Experimento 5.6
- `results_expanding_cv.csv` + `results_expanding_cv_fold_agg.csv` — Experimento 5.8
- `results_vale3_deepdive.csv` — Experimento 5.9
- `results_ablation.csv` + `ablation_summary.csv` — Experimento 5.10
