# Análise dos Resultados — Etapa 5: Ajuste de Threshold

**Objetivo:** Buscar o threshold ótimo de classificação no conjunto de validação para corrigir
as previsões degeneradas observadas na Etapa 4 (todos os modelos exceto o Transformer previam
100% de uma única classe com threshold 0.5).

**Método:** Varredura de thresholds de 0.10 a 0.90, selecionando o que maximiza F1-score na validação (176–180 amostras), depois avaliando no teste (177–182 amostras).

---

## Thresholds Ótimos Encontrados

| Modelo | Threshold Ótimo | Observação |
|--------|:-:|---|
| BiLSTM Original (2L/128h) | 0.10 | Mínimo da faixa — modelo colapsado |
| BiLSTM Reduzido (1L/64h) | 0.10 | Mínimo da faixa — modelo colapsado |
| Transformer | 0.44 | Próximo de 0.5 — modelo melhor calibrado |
| XGBoost | 0.10 | Mínimo da faixa — probabilidades muito baixas |

O fato de 3 dos 4 modelos terem threshold ótimo no limite inferior (0.10) é um **sinal de alerta**: indica que esses modelos não aprenderam a discriminar entre as classes de forma útil. Quando o melhor threshold é 0.10, o modelo basicamente prevê "Sobe" para quase tudo — e isso "funciona" apenas porque 69% do teste é de fato "Sobe".

---

## Resultados no Teste: Default (0.5) vs Otimizado

### BiLSTM Original (2L/128h)

| Métrica | t=0.50 | t=0.10 | Δ |
|---------|:------:|:------:|:-:|
| ROC-AUC | 0.465 | 0.465 | — |
| Accuracy | 30.5% | 69.5% | +39.0pp |
| F1 (Sobe) | 0.000 | 0.820 | +0.820 |

A melhora aparente é **ilusória**: o modelo passou de prever 100% "Desce" para prever 100% "Sobe". Ambos os extremos são degenerados. O ROC-AUC de 0.465 (abaixo do acaso) confirma que o modelo não tem poder discriminativo. A "accuracy" de 69.5% é simplesmente a proporção da classe majoritária.

### BiLSTM Reduzido (1L/64h)

| Métrica | t=0.50 | t=0.10 | Δ |
|---------|:------:|:------:|:-:|
| ROC-AUC | 0.342 | 0.342 | — |
| Accuracy | 30.5% | 69.5% | +39.0pp |
| F1 (Sobe) | 0.000 | 0.820 | +0.820 |

Mesmo padrão do Original. ROC-AUC de 0.342 é **pior que o acaso** — o modelo aprendeu padrões anti-correlacionados. Nenhum threshold corrige isso.

### Transformer

| Métrica | t=0.50 | t=0.44 | Δ |
|---------|:------:|:------:|:-:|
| ROC-AUC | 0.636 | 0.636 | — |
| Accuracy | 30.5% | 41.2% | +10.7pp |
| F1 (Sobe) | 0.000 | 0.268 | +0.268 |

O Transformer é o único modelo com threshold razoável (0.44 vs 0.50), indicando calibração melhor. No entanto, nesta execução o desempenho caiu em relação à Etapa 4 (AUC 0.636 vs 0.709). Isso revela **instabilidade**: redes neurais com poucos dados variam significativamente entre execuções.

### XGBoost

| Métrica | t=0.50 | t=0.10 | Δ |
|---------|:------:|:------:|:-:|
| ROC-AUC | 0.670 | 0.670 | — |
| Accuracy | 30.8% | 69.2% | +38.5pp |
| F1 (Sobe) | 0.000 | 0.818 | +0.818 |

O XGBoost mantém o melhor AUC (0.670), consistente com a Etapa 4. Porém, com threshold 0.10, ele prevê "Sobe" para quase todas as amostras — a accuracy de 69.2% não reflete poder preditivo real.

---

## Tabela Comparativa Final

| Modelo | ROC-AUC | Threshold | Accuracy | F1 (Sobe) | Poder Real? |
|--------|:-------:|:---------:|:--------:|:---------:|:-----------:|
| BiLSTM Original | 0.465 | 0.10 | 69.5% | 0.820 | Não |
| BiLSTM Reduzido | 0.342 | 0.10 | 69.5% | 0.820 | Não |
| Transformer | 0.636 | 0.44 | 41.2% | 0.268 | Parcial |
| **XGBoost** | **0.670** | 0.10 | 69.2% | 0.818 | Parcial |

---

## Diagnóstico

### Por que o ajuste de threshold não resolveu o problema?

O ajuste de threshold é útil quando o modelo **discrimina bem mas está mal calibrado** — ou seja, quando o ROC-AUC é alto mas o threshold padrão não funciona. Neste caso:

1. **LSTMs (AUC < 0.50):** Não discriminam. Nenhum threshold pode salvar um modelo que não aprendeu.

2. **XGBoost (AUC 0.670):** Discrimina parcialmente, mas as probabilidades estão todas concentradas em uma faixa estreita abaixo de 0.5. O threshold de 0.10 é um "hack" que converte isso em previsão majoritária — não é uma solução genuína.

3. **Transformer (AUC 0.636):** Discrimina parcialmente e tem a melhor calibração (threshold 0.44). Porém, a instabilidade entre execuções (AUC variando de 0.636 a 0.709) indica que o modelo é sensível à inicialização.

### O problema fundamental

O sinal de sentimento FinBERT é **fraco e ruidoso** para prever a direção do preço em 21 dias úteis. Possíveis razões:

- **Defasagem temporal:** o sentimento de hoje pode já estar precificado quando o horizonte de 21 dias é avaliado
- **Causalidade reversa:** notícias podem ser reativas ao preço, não preditivas
- **Poucas features:** 5 dimensões de sentimento diário fornecem pouca informação por amostra
- **Volume de dados:** ~1.200 amostras para treino é limitado para detectar padrões sutis

---

## Conclusões

1. **O ajuste de threshold não substituiu poder discriminativo.** Modelos com AUC < 0.5 continuam inúteis independente do threshold.

2. **XGBoost permanece o modelo mais estável** (AUC 0.670 consistente entre execuções), mas com discriminação insuficiente para uso prático.

3. **Transformer mostra potencial mas instável** — precisa de mais dados ou regularização para estabilizar.

4. **Accuracy com threshold otimizado é enganosa** — quando 3/4 modelos têm threshold ótimo em 0.10, a "melhoria" é apenas previsão da classe majoritária.

5. **Próximos passos mais promissores que ajuste de threshold:**
   - Combinar sentimento FinBERT + embeddings Ollama como features conjuntas
   - Reduzir horizonte de previsão (5 ou 10 dias) para capturar impacto imediato do sentimento
   - Expandir para multi-ticker (PETR4, VALE3) para aumentar o volume de treino
   - Usar ensemble (XGBoost + Transformer) para combinar estabilidade com capacidade
