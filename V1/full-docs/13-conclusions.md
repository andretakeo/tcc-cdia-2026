# 13 — Conclusions, Limitations, and Future Work

## Contributions

The thesis offers three contributions:

### 1. Replicable Technical Pipeline

A complete, versioned system for:
- News collection from InfoMoney (WordPress API)
- Sentiment extraction via FinBERT-PT-BR
- Integration with Yahoo Finance market data
- Binary classification with 6 model architectures

### 2. Empirical Demonstration of Single-Window Bias

The inversion from AUC 0.709 (single window) to ~0.51 (multi-fold) is documented with:
- 1,500+ model runs
- 13 experiments
- Formal statistical support (Wilcoxon tests, bootstrap CIs, power analysis)
- Coverage across 3 stocks and 2 horizons

This is the most detailed demonstration of this phenomenon on Brazilian financial data.

### 3. Minimum Evaluation Protocols

Six practices proposed for financial ML research:

1. **Report bootstrap CI** (95%, 1,000 resamples) on ROC-AUC
2. **Train with ≥10 seeds** and report mean ± std
3. **Use expanding-window CV** instead of single split
4. **Compare against autoregressive baseline** (XGBoost on price features)
5. **Monitor prediction distributions** and confusion matrices
6. **Audit validation-test AUC correlation** for non-stationarity detection

## Self-Correction Reflection

The experience of building a pipeline, obtaining a strong-looking result (AUC = 0.709), and then systematically dismantling it was the most valuable lesson. Confirmation bias operated invisibly: the absence of confidence intervals, the single seed, and the single split didn't seem problematic because the result "made sense."

The protocols proposed are not original — multiple seeds, temporal CV, and autoregressive baselines are explicit recommendations in López de Prado (2018). The contribution is showing, in a concrete case with Brazilian data, the magnitude of damage caused by ignoring them: an inversion of 0.709 → ~0.51, turning a positive conclusion into a negative one.

## Limitations

1. **Single news source** (InfoMoney): Editorial bias toward retail investors, heterogeneous coverage across tickers (2,572 articles for ITUB4 vs 1,525 for VALE3)

2. **Only 3 large-cap stocks**: Limits external validity to similar B3 equities

3. **Horizons restricted to h=5 and h=21**: Intraday or very long horizons may behave differently

4. **Specific sentiment representation**: 5 aggregated daily features from FinBERT-PT-BR. Other representations (article-level, sentence-level, multi-model ensembles) could yield different results

5. **`pos_weight` bug in Stages 3-4**: The BiLSTM code computes `pos_weight` but never applies it (uses `BCELoss` instead of `BCEWithLogitsLoss`). Affects only BiLSTM results in Chapters 3-4, which were already near chance

6. **PCA leakage in Stage 3**: PCA fitted on full dataset (train+val+test), leaking variance structure. Does not affect Stage 4+ results (no PCA)

7. **Forward-fill single-ticker test**: The ffill sensitivity analysis covers only ITUB4. PETR4 and VALE3 have different news coverage densities, so the bias magnitude could differ

8. **Power limitations**: With test sets of 60–177 samples, real effects smaller than Δ AUC ≈ 0.05 are undetectable. The conclusion is "no detectable effect at this sample size," not "no effect"

## Future Work

1. **Diversify text sources**: CVM filings, analyst reports, social media (Twitter/X), multiple news portals. Stage 8 explored this but was not completed

2. **Test collapse-resistant architectures**: Ensemble methods, smaller Transformers with stronger regularization, or architectures that degrade gracefully in low-data regimes

3. **Generalize to other markets**: Apply the same pipeline and validation protocols to non-Brazilian equities to test whether the null result is market-specific

4. **Formal study of validation-test anticorrelation**: The negative correlation between validation AUC and test AUC (Experiment 7) deserves theoretical treatment as a marker of non-stationarity

5. **Alternative metrics**: Investigate balanced accuracy, Matthews correlation coefficient, or profit-based metrics as alternatives to AUC in class-imbalanced settings
