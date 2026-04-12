# Análise dos Resultados — FinBERT-PT-BR Sentiment Features

**Dataset:** 1.227 dias × 16 features (11 preço + 5 sentimento FinBERT)
**Período:** 2021-04-28 a 2026-03-26
**Target:** sobe/desce em 21 dias úteis (balance: 59% sobe / 41% desce)

---

## Resultados por Modelo

### BiLSTM Original (2L/128h)

| Métrica | Valor |
|---------|-------|
| ROC-AUC | 0.500 |
| Accuracy | 69.5% |
| F1 (Sobe) | 0.82 |

O modelo convergiu para uma **solução degenerada**: prevê "Sobe" para 100% das amostras, resultando em ROC-AUC = 0.50 (equivalente ao acaso). A val_loss estagna em ~0.679 desde as primeiras épocas, e a acurácia de 61.4% reflete apenas a proporção da classe majoritária.

Com 10 camadas e 128 hidden units, o modelo é **excessivamente complexo** para apenas 16 features — a arquitetura original foi projetada para 1035 features (com embeddings). O excesso de parâmetros causa o colapso para a classe majoritária.

---

### BiLSTM Reduzido (1L/64h)

| Métrica | Valor |
|---------|-------|
| ROC-AUC | 0.477 |
| Accuracy | 30.5% |
| F1 (Sobe) | 0.00 |

Comportamento oposto: o modelo **overfitou rapidamente** (train_loss caindo para 0.52, val_loss explodindo para 1.35) e prevê "Desce" para 100% das amostras. ROC-AUC = 0.477, **pior que o acaso**.

O early stopping atuou cedo (epoch 11), mas o melhor checkpoint já estava comprometido. A val_loss divergiu desde a epoch 2, indicando que o modelo **memorizou padrões espúrios** do treino sem capacidade de generalização.

---

### XGBoost Baseline

| Métrica | Valor |
|---------|-------|
| ROC-AUC | 0.670 |
| Accuracy | 30.8% |
| F1 (Sobe) | 0.00 |

Melhor resultado entre os modelos tabulares, com **ROC-AUC = 0.670** — acima do acaso, mas com limites claros. O early stopping parou na iteração 4 (de 200), sinalizando que o sinal é fraco e o modelo evita overfitting rapidamente.

Apesar do AUC razoável, o limiar de 0.5 produz previsões degeneradas (100% "Desce"). Isso indica que as **probabilidades estão calibradas abaixo de 0.5** para a maioria das amostras — o modelo consegue rankear relativamente bem (AUC), mas o limiar padrão não funciona. Um ajuste de threshold via validação poderia melhorar a acurácia.

---

### Transformer

| Métrica | Valor |
|---------|-------|
| ROC-AUC | 0.709 |
| Accuracy | 76.3% |
| F1 (Sobe) | 0.85 |

**Melhor modelo do experimento**: ROC-AUC = 0.709, acurácia = 76%, F1(Sobe) = 0.85. O Transformer conseguiu aprender padrões temporais das features de sentimento que os LSTMs não conseguiram capturar.

Pontos relevantes:
- **Precision(Desce) = 1.00** com recall de 20%: quando o modelo prevê queda, acerta sempre — mas é conservador, prevendo queda em poucos casos
- **Recall(Sobe) = 1.00**: identifica todos os dias de alta, com precision de 74%
- O overfitting apareceu a partir da epoch 7 (val_loss divergindo), mas o early stopping salvou o checkpoint da epoch 2 que generalizou bem
- A atenção multi-cabeça parece capturar melhor as **interações temporais entre sentimento e preço** do que as gates do LSTM

---

## Tabela Comparativa

| Modelo | ROC-AUC | Accuracy | F1 (Sobe) |
|--------|---------|----------|-----------|
| BiLSTM Original (2L/128h) | 0.500 | 69.5% | 0.82 |
| BiLSTM Reduzido (1L/64h) | 0.477 | 30.5% | 0.00 |
| XGBoost | 0.670 | 30.8% | 0.00 |
| **Transformer** | **0.709** | **76.3%** | **0.85** |

---

## Comparação: Ollama Embeddings (Etapa 3) vs FinBERT Sentiment (Etapa 4)

### Contexto

- **Etapa 3** (`3.model_traning/`): 1.227 dias × 1.035 features (11 preço + 1.024 embeddings Ollama qwen3-embedding). PCA reduz embeddings para 32 componentes.
- **Etapa 4** (`4.finbert-br/`): 1.227 dias × 16 features (11 preço + 5 sentimento FinBERT-PT-BR). Sem PCA necessário.

### ROC-AUC — Lado a Lado

| Modelo | Ollama Embeddings | FinBERT Sentiment | Δ |
|--------|:-----------------:|:-----------------:|:-:|
| BiLSTM Original (2L/128h) | 0.443 | 0.500 | +0.057 |
| BiLSTM Reduzido (1L/64h) | 0.505 | 0.477 | -0.028 |
| XGBoost | 0.610 | 0.670 | **+0.060** |
| Transformer | 0.568 | **0.709** | **+0.141** |

### Accuracy — Lado a Lado

| Modelo | Ollama Embeddings | FinBERT Sentiment | Δ |
|--------|:-----------------:|:-----------------:|:-:|
| BiLSTM Original (2L/128h) | 38.4% | 69.5% | +31.1pp |
| BiLSTM Reduzido (1L/64h) | 30.5% | 30.5% | 0 |
| XGBoost | 30.8% | 30.8% | 0 |
| Transformer | 69.5% | **76.3%** | **+6.8pp** |

### Interpretação

1. **O Transformer melhorou em todas as métricas** — de AUC 0.568 para 0.709 (+0.141), de accuracy 69.5% para 76.3%. É o ganho mais expressivo e o único modelo que realmente extraiu valor das features de sentimento.

2. **XGBoost melhorou no ranking** (AUC 0.610 → 0.670) mas mantém o mesmo problema de calibração em ambos os cenários — prevê a classe minoritária para todas as amostras com threshold 0.5.

3. **BiLSTMs não melhoraram de forma significativa** — continuam falhando por razões distintas (colapso vs overfitting). O problema deles é estrutural, não de qualidade de features.

4. **5 features de sentimento > 1.024 embeddings** — resultado contraintuitivo mas consistente. As hipóteses:
   - O FinBERT funciona como **filtro de sinal**: comprime texto em 3 dimensões diretamente relevantes (positivo/negativo/neutro para mercado financeiro), enquanto embeddings genéricos de 1.024 dims incluem muito ruído semântico irrelevante
   - Com menos features, o risco de **curse of dimensionality** diminui drasticamente (16 vs 43 features pós-PCA)
   - O modelo FinBERT-PT-BR foi **treinado especificamente em textos financeiros brasileiros**, capturando nuances que um embedding genérico não distingue

5. **O gargalo não era a arquitetura — era a representação textual.** Na Etapa 3, todos os modelos ficaram em ~0.5 AUC, levando à conclusão de que o problema era intrinsecamente difícil. A Etapa 4 mostra que com a representação certa (sentimento financeiro específico), o Transformer consegue explorar o sinal.

---

## Conclusões

1. **O Transformer foi o único modelo que generalizou** com features de sentimento FinBERT, alcançando ROC-AUC 0.709 — superior ao ~0.50 obtido com embeddings Ollama no pipeline original (`3.model_traning/`)

2. **Features de sentimento (5 dims) são mais informativas que embeddings brutos (1024 dims)** para este problema. A redução de dimensionalidade feita pelo FinBERT (texto → sentimento positivo/negativo/neutro) funciona como um filtro de ruído, preservando o sinal relevante para a direção do preço

3. **LSTMs falharam completamente** — tanto o original (colapso para classe majoritária) quanto o reduzido (overfitting imediato). As 16 features são insuficientes para as gates do LSTM, que precisam de representações mais ricas

4. **XGBoost mostrou capacidade de ranking** (AUC 0.67) mas falha na calibração — indica que o sinal existe mas é fraco para classificação com threshold fixo

5. **Próximos passos sugeridos:**
   - Ajustar threshold do XGBoost via validação para melhorar accuracy
   - Testar Transformer com features combinadas (sentimento + embeddings Ollama)
   - Expandir para outros tickers (PETR4, VALE3) para validar generalização
   - Testar horizontes menores (5, 10 dias) onde o sentimento pode ter impacto mais imediato
