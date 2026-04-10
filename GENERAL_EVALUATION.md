General Scientific Critical Evaluation — TCC CDIA 2026

Research question: Can Brazilian financial news improve stock price direction prediction (up/down) for B3 equities?

---

1. Strengths  


Methodological self-correction (exceptional)

The most impressive aspect of this work is Chapter 5's systematic demolition of Chapter 4's headline result. This is rare in undergraduate work and uncommon even in published research. The progression from a single-window AUC of 0.709 to a methodologically correct
expanding-window AUC of ~0.51 is documented with 1,435 model runs, statistical tests, and transparent reporting. This alone elevates the thesis above typical ML-for-finance papers.

Experimental rigor in Chapter 5

- Bootstrap confidence intervals (1,000 resamples)
- Multi-seed variance analysis (20-40 seeds)
- Expanding-window CV (up to 52 folds)
- Wilcoxon signed-rank tests with paired comparisons
- Ablation study isolating sentiment contribution
- Practical validation via backtest (Sharpe ratio, drawdown)

Reproducibility

Code, notebooks, CSVs, and figures are versioned. Analysis is traceable from raw data to final conclusion. This is good scientific practice.

---

2. Concerns

Critical Issues

2.1 — The Chapters 3-4 evaluation protocol is a known anti-pattern, not a novel finding.

Single-window evaluation in non-stationary financial time series has been criticized extensively in the literature (e.g., Bailey et al. 2014 "Pseudo-Mathematics and Financial Charlatanism"; Loppez de Prado 2018 "Advances in Financial Machine Learning"). Chapter 5
frames the discovery as if the single-window flaw was unexpected. For a thesis defense, the committee may ask: why was this protocol adopted initially if the literature already warns against it? A brief literature review of evaluation protocols in financial ML would
strengthen the positioning — currently, the references section appears thin on methodological precedents.

2.2 — The "dumb baseline" is not dumb enough.

The baseline uses return, lag_1, lag_5, Volume, std21 with XGBoost — this is already a reasonable feature-engineered model. A truly naive baseline would be:

- Majority class predictor (always predict "up")
- Random coin flip weighted by class prior
- Persistence forecast (tomorrow = today)

The current "dumb baseline" is actually a competent autoregressive model. This matters because the thesis claims "sentiment adds nothing beyond a trivial baseline" — but the baseline isn't trivial. The correct framing is: sentiment adds nothing beyond price-derived
features, which is a weaker (but still important) claim.

2.3 — Confounding between model architecture and feature contribution.

Chapters 3-4 change two variables simultaneously: text representation (Ollama -> FinBERT) AND dimensionality (1024 -> 5). The conclusion "domain-specific representation beats generic embeddings" is confounded by the curse of dimensionality. An XGBoost on 5 random
features might also beat XGBoost on 1024 PCA-compressed embeddings simply due to reduced noise, not domain specificity.

Chapter 5's ablation (Experiment 8) partially addresses this for PRICE vs SENT vs PRICE+SENT, but the original Chapter 3 vs 4 comparison remains confounded.

Important Issues

2.4 — Small test set sizes inflate AUC variance.

Test sets of 150-177 samples are acknowledged as a limitation but not adequately quantified. With ~60% class imbalance and N=177, the standard error of AUC under the null is approximately 0.04-0.06. The thesis correctly uses bootstrap CIs in Chapter 5 but doesn't
discuss whether the expanding-window folds (90-day windows, ~60-90 test samples per fold) have adequate statistical power to detect meaningful effects.

2.5 — The 21-day horizon choice is weakly motivated.

The thesis tests h=5 and h=21 days and finds h=5 is better, attributing this to "news have short-term impact." But the initial choice of h=21 is not justified — why 21 specifically? Was this from literature, or arbitrary? If arbitrary, the comparison h=5 vs h=21 is
a two-point sample from a continuous space. A sweep over multiple horizons (1, 2, 5, 10, 21, 42 days) would strengthen the "short-term effect" claim considerably.

2.6 — Forward-fill of sentiment on non-news days introduces look-ahead risk.

Section 4.3 states: "em dias de pregão sem notícias publicadas, as features de sentimento são preenchidas com forward-fill." If a news article is published after market close on day t, its sentiment fills day t, but the price effect may only appear on day t+1. The
timestamp granularity (daily vs intraday) is not discussed. This is a potential look-ahead bias that could inflate apparent predictive power of sentiment features.

2.7 — Single news source limits external validity.

InfoMoney alone covers a specific editorial lens. The thesis acknowledges this but doesn't discuss how this might bias results — e.g., InfoMoney may publish delayed reactions to events already priced in by institutional investors using Bloomberg/Reuters.

Minor Issues

2.8 — The Stage 7 comparative analysis reports different "best models" than Chapters 3-4.

Chapter 4 reports Transformer AUC = 0.709 for Stage 4, but the Stage 7 re-evaluation (ANALISE_COMPARATIVA_FINAL.md) reports Random Forest as Stage 4's best model at AUC = 0.559. The discrepancy is explained by seed sensitivity, but a reader encountering both
documents may be confused. A reconciliation table would help.

2.9 — TCN's superiority (Stage 5b, AUC 0.643) was not subjected to Chapter 5-level scrutiny.

The best practical result (TCN [32,32] with engineered features) was evaluated in Stage 7's comparative analysis but was not put through the multi-seed, expanding-window, ablation protocol of Chapter 5. If this is the thesis's best configuration, it deserves the
same rigor. Its current validation may suffer the same single-window artifacts.

2.10 — Effect size language could be more precise.

Phrases like "the gap is huge" or "dramatic inversion" are used where quantified effect sizes (Cohen's d, rank-biserial correlation) would be more informative and defensible.

---

3. Logical and Statistical Assessment

┌─────────────────────────────┬──────────────┬─────────────────────────────────────────────────┐
│ Criterion │ Rating │ Notes │
├─────────────────────────────┼──────────────┼─────────────────────────────────────────────────┤
│ Research question clarity │ Good │ Well-defined, falsifiable │
├─────────────────────────────┼──────────────┼─────────────────────────────────────────────────┤
│ Internal validity (Ch. 3-4) │ Weak │ Single window, single seed, no CI │
├─────────────────────────────┼──────────────┼─────────────────────────────────────────────────┤
│ Internal validity (Ch. 5) │ Strong │ Multi-fold, multi-seed, statistical tests │
├─────────────────────────────┼──────────────┼─────────────────────────────────────────────────┤
│ External validity │ Limited │ 1 source, 3 stocks, 1 market │
├─────────────────────────────┼──────────────┼─────────────────────────────────────────────────┤
│ Statistical reporting │ Good (Ch. 5) │ Bootstrap CI, Wilcoxon, effect sizes │
├─────────────────────────────┼──────────────┼─────────────────────────────────────────────────┤
│ Confound control │ Moderate │ Architecture-representation confound unresolved │
├─────────────────────────────┼──────────────┼─────────────────────────────────────────────────┤
│ Reproducibility │ Strong │ Code + data + notebooks versioned │
├─────────────────────────────┼──────────────┼─────────────────────────────────────────────────┤
│ Appropriate conclusions │ Strong │ Claims proportional to evidence in Ch. 5 │
└─────────────────────────────┴──────────────┴─────────────────────────────────────────────────┘

---

4. Overall Assessment

This is a strong undergraduate thesis with an unusually honest methodological arc. The core contribution — demonstrating how single-window evaluation creates illusory results in financial ML — is well-supported by 1,435 model runs and formal statistical tests. The
work transitions from a naive positive result (AUC 0.709) to a rigorously negative one (sentiment adds +0.003, p=0.49), which requires intellectual courage.

Key risks for defense:

1. Committee may ask why known evaluation pitfalls weren't avoided from the start
2. The TCN/Stage 5b result (the "best practical result") lacks Chapter 5-level validation
3. The forward-fill timestamp issue could be probed

Recommendations:

1. Add a brief literature review of evaluation protocols in financial ML to justify the progression from naive to rigorous
2. Subject the TCN Stage 5b configuration to at least multi-seed analysis
3. Clarify timestamp handling for news-to-price alignment
4. Consider renaming "dumb baseline" to "autoregressive baseline" — it's not dumb

The thesis as written is publishable as a methodological case study, which is a strong outcome for a TCC.
