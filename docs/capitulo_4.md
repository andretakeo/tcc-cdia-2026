# Capítulo 4 — Sentimento Financeiro Específico via FinBERT-PT-BR

> **Nota de leitura.** Este capítulo apresenta um *pipeline* de extração e treinamento que produz, sob o protocolo de avaliação adotado nos capítulos anteriores (*walk-forward split* único 70/15/15), o resultado quantitativamente mais forte da dissertação até este ponto. O Capítulo 5 investiga em detalhe a robustez metodológica desse resultado e revisa substancialmente sua interpretação. O leitor é encorajado a ler os Capítulos 4 e 5 como uma unidade.

## 4.1 Hipótese

Os resultados do Capítulo 3 mostraram que nenhum dos quatro modelos avaliados (BiLSTM original, BiLSTM reduzido, XGBoost, Transformer) supera consistentemente o acaso quando treinados sobre o conjunto combinado de 11 features de preço e 1.024 *embeddings* densos genéricos extraídos via Ollama (`qwen3-embedding:4b`). O melhor resultado foi o XGBoost com ROC-AUC = 0.610. A interpretação inicial atribuiu o desempenho fraco a uma de duas causas:

- **(A) Hipótese de eficiência de mercado** — a previsão de direção a 21 dias úteis é intrinsecamente difícil em mercados líquidos como o brasileiro, e nenhum modelo melhoraria significativamente o resultado.
- **(B) Hipótese de representação textual** — os *embeddings* genéricos contêm muito ruído semântico irrelevante para o domínio financeiro. Uma representação compacta e específica do domínio poderia ser mais informativa.

O Capítulo 4 testa a hipótese (B) substituindo os 1.024 *embeddings* genéricos por **5 *features* de sentimento financeiro** extraídas com um modelo BERT especializado em textos financeiros em português brasileiro: o **FinBERT-PT-BR**. A escolha de uma representação de baixíssima dimensionalidade — 5 dimensões em vez de 1.024 — é deliberada: se uma compressão tão agressiva, mas informada por conhecimento de domínio, supera os *embeddings* genéricos, isso constitui evidência de que o gargalo do Capítulo 3 era de representação, não de capacidade dos modelos.

## 4.2 O modelo FinBERT-PT-BR

**FinBERT-PT-BR** é um modelo BERT pré-treinado e *fine-tuned* em corpora de notícias e relatórios financeiros em português brasileiro. Ele recebe texto bruto como entrada e produz, para cada *input*, três *logits* correspondentes às classes:

- `POSITIVO` — sentimento financeiro positivo (notícia favorável a desempenho de preço)
- `NEGATIVO` — sentimento financeiro negativo (notícia desfavorável)
- `NEUTRO` — sentimento neutro ou irrelevante para movimentação de preço

Diferentemente de modelos de sentimento genéricos (treinados em redes sociais ou avaliações de produtos), o FinBERT-PT-BR foi calibrado para vocabulário e contextos financeiros específicos: termos como "déficit", "rentabilidade", "guidance", "downgrade" e "ação preferencial" recebem peso semântico adequado ao domínio.

O modelo é carregado localmente via `transformers.AutoModelForSequenceClassification` a partir do diretório `4.finbert-br/FinBERT-PT-BR/`, sem dependência de chamadas externas durante a inferência.

## 4.3 Pipeline de extração de sentimento

A extração de *features* de sentimento segue os seguintes passos, implementados em `4.finbert-br/index.ipynb`:

1. **Carregamento dos artigos** coletados na Etapa 1 (Capítulo 2): `itub4_noticias.json` (2.572 artigos), `petr4_noticias.json` (1.775), `vale3_noticias.json` (1.525), totalizando **5.872 artigos** processados.

2. **Inferência em *batch*** com 32 artigos por *batch*. O *input* de cada artigo é a concatenação `título + resumo`, truncada em 512 *tokens* — o limite do BERT. Este *input* é mais compacto e foco-específico do que o texto completo do artigo, minimizando dispersão semântica.

3. **Saída por artigo** — classe predita (POS/NEG/NEU) e *logits* brutos `[pos, neg, neu]`. Os *logits* (anteriores ao *softmax*) preservam mais informação que a classe discreta e são preferíveis como entrada para modelos *downstream*.

4. **Persistência por artigo** em `{itub4,petr4,vale3}_noticias_sentiment.json`.

5. **Agregação diária** — para cada dia em que houve publicação de notícias, são calculadas:
   - `n_articles` — contagem de artigos do dia
   - `mean_logit_pos`, `mean_logit_neg`, `mean_logit_neu` — média dos *logits* dos artigos do dia
   - `mean_sentiment` — média da classe discreta predita (codificação ordinal: 0=NEG, 1=NEU, 2=POS)

   Resultado: **5 features por dia**, persistidas em `{itub4,petr4,vale3}_daily_sentiment.csv`.

6. **Junção temporal com *features* de preço** — *left join* com as 11 *features* OHLCV/técnicas geradas no Capítulo 2, indexadas por data. Em dias de pregão sem notícias publicadas, as *features* de sentimento são preenchidas com *forward-fill* (propagação do último valor disponível). Resultado final: **dataset de 1.207 dias × 16 features** (11 de preço + 5 de sentimento), cobrindo o período de 2021-04 a 2026-03 para ITUB4.

## 4.4 Protocolo de treinamento

Os mesmos quatro modelos do Capítulo 3 são retreinados sobre o novo *dataset*:

- **BiLSTM Original** — 2 camadas, 128 *hidden units*, 30% *dropout*
- **BiLSTM Reduzido** — 1 camada, 32 *hidden units*, 50% *dropout*
- **XGBoost Baseline** — 300 árvores, profundidade 4, *learning rate* 0.05
- **Transformer** — 2 camadas de atenção, 4 cabeças, `d_model=64`, *positional encoding* sinusoidal, *mean pooling* temporal

A arquitetura, otimizador (Adam, `lr=1e-3`, `weight_decay=1e-4`), função de perda (`BCEWithLogitsLoss` com `pos_weight` para mitigar desbalanceamento), *early stopping* (paciência 10), tamanho de janela temporal (30 dias) e *split* cronológico walk-forward (70% treino / 15% validação / 15% teste, sem *shuffle*) são idênticos ao Capítulo 3, com uma única diferença: como a dimensionalidade já é baixa (16 *features* contra 1.035 do Capítulo 3), **não é aplicado PCA**.

A semente de inicialização de PyTorch é fixada (`torch.manual_seed(42)`) para reprodutibilidade aparente. O Capítulo 5 documenta em detalhe por que esta única fixação é insuficiente.

O alvo binário é definido como `1` se `Close[t+21] > Close[t]`, caso contrário `0`, resultando em desbalanceamento aproximado de 59% "Sobe" / 41% "Desce" no conjunto de treino do ITUB4.

## 4.5 Resultados sob *walk-forward split* único

A tabela abaixo sumariza o ROC-AUC obtido pelos quatro modelos no conjunto de teste, comparando o resultado da Etapa 3 (1.024 *embeddings* genéricos do Ollama, após PCA para 32 dimensões) com o da Etapa 4 (5 *features* de sentimento FinBERT-PT-BR):

| Modelo | AUC Etapa 3 (1.024 dim) | AUC Etapa 4 (5 dim sentimento) | Δ |
|---|:---:|:---:|:---:|
| BiLSTM Original | 0.443 | 0.500 | +0.057 |
| BiLSTM Reduzido | 0.505 | 0.477 | −0.028 |
| XGBoost | 0.610 | 0.670 | +0.060 |
| **Transformer** | 0.568 | **0.709** | **+0.141** |

O Transformer atinge **ROC-AUC = 0.709**, o melhor resultado obtido até este ponto da dissertação. Métricas adicionais para o Transformer:

- **Acurácia**: 76.3% (177 amostras de teste)
- **F1 (Sobe)**: 0.85 (recall 1.00, precisão 0.74)
- **F1 (Desce)**: 0.34 (recall 0.20, **precisão 1.00**)
- **Matriz de confusão**: das 11 vezes em que o modelo prevê "Desce", 11/11 estão corretas; mas o modelo só faz 11 previsões "Desce" em 177 amostras

O XGBoost também melhora substancialmente (0.610 → 0.670, Δ = +0.060), corroborando a leitura de que a representação compacta de domínio é mais informativa que os *embeddings* genéricos para modelos clássicos. O BiLSTM original sai do estado patológico (0.443, abaixo do acaso) para o limite do acaso (0.500), e o BiLSTM reduzido permanece próximo do acaso. A intuição imediata é que modelos sequenciais profundos sofrem por excesso de capacidade no regime de baixo volume de dados disponível (~800 amostras de treino), enquanto modelos com viés indutivo mais forte ou com regularização tabular (XGBoost, Transformer com janelamento + atenção temporal) conseguem extrair sinal.

## 4.6 Leitura inicial e o gancho para o Capítulo 5

Sob a leitura padrão da literatura, os resultados acima sustentariam três conclusões:

1. **A representação textual importa mais que a arquitetura.** Trocando apenas a representação (1.024 *embeddings* → 5 *features* de sentimento) e mantendo o mesmo Transformer, o AUC sobe de 0.568 para 0.709. Esta é uma diferença grande.

2. **5 *features* específicas superam 1.024 *features* genéricas.** A redução agressiva de dimensionalidade, quando feita por um modelo especializado em domínio, atua como filtro de sinal — elimina ruído sem destruir informação preditiva.

3. **O melhor *pipeline* é Transformer + FinBERT + horizonte de 21 dias.** Esta seria a configuração recomendada como contribuição da dissertação.

Três observações qualitativas, contudo, justificam um exame metodológico adicional antes de adotar essas conclusões como definitivas:

- **A matriz de confusão do Transformer é altamente assimétrica.** Das 177 amostras de teste, o modelo prevê "Sobe" em 166 (94%) e "Desce" em apenas 11 (6%). A acurácia de 76.3% reflete em grande parte a proporção da classe majoritária no teste (~69%); a contribuição líquida do modelo sobre um preditor constante é de aproximadamente 7 pontos percentuais.

- **Os intervalos de confiança não foram calculados.** Em conjuntos de teste de tamanho moderado (~150–200 amostras), o desvio-padrão típico do AUC é da ordem de 0.04–0.08. Diferenças menores que esse valor estão dentro do ruído de estimação, e a diferença de 0.099 entre as Etapas 3 e 4 do Transformer está apenas marginalmente acima desse limiar.

- **Apenas uma semente foi usada.** Modelos profundos em regimes de baixo volume de dados são conhecidos por exibir alta variância de inicialização. Os resultados acima representam um único ponto amostrado da distribuição induzida pela semente fixada em 42.

Estas observações motivam a investigação metodológica sistemática conduzida no Capítulo 5, que aplica progressivamente *bootstrap* CI, treinamento *multi-seed*, validação cruzada *expanding-window* multi-fold, *deep-dive* estatístico e *ablation* de *features*. Os achados desse capítulo revisam substancialmente a interpretação dos resultados aqui apresentados.

## 4.7 Materiais e código

O *pipeline* completo desta etapa está em `4.finbert-br/`:

- `index.ipynb` — extração de sentimento (carregamento do modelo, inferência em *batch*, agregação diária)
- `model_training.ipynb` — retreinamento dos 4 modelos com as *features* de sentimento
- `FinBERT-PT-BR/` — pesos do modelo pré-treinado
- `{itub4,petr4,vale3}_daily_sentiment.csv` — *features* diárias de sentimento por ativo
- `{itub4,petr4,vale3}_noticias_sentiment.json` — sentimento por artigo
- `ANALISE_RESULTADOS.md` — análise detalhada modelo a modelo
- `lstm_results.png`, `transformer_results_finbert.png`, `roc_comparison_finbert.png`, `xgboost_roc_finbert.png` — visualizações de treinamento

Os volumes de artigos processados e as estatísticas agregadas por ativo estão registradas no notebook `index.ipynb` e nos arquivos `*_daily_sentiment.csv`.
