# PRD: Quantitative Research & Backtest Platform (V2)

## 1. Executive Summary

The **Quantitative Research & Backtest Platform (V2)** is a production-grade analytical environment designed to validate the predictive power of financial news sentiment on Brazilian stock prices (specifically ITUB4). Transitioning from a batch-processed exploratory phase (V1), the V2 platform implements strict methodological shielding to eliminate statistical biases and provide a real-time, event-driven research lab.

---

## 2. Problem Statement

Traditional financial NLP research often suffers from:

1.  **Look-Ahead Bias:** Evening news affecting same-day closing prices in models.
2.  **Noise Drowning:** High-volume, low-impact news diluting significant market signals.
3.  **Static Evaluation:** Single-window splits failing to account for non-stationary market regimes.
4.  **Baseline Mirage:** High accuracy claims without comparison to naive heuristics (Inertia/Majority).

---

## 3. Goals & Objectives

- **Scientific Validation:** Provide a "Scientific Rigor Score" comparing advanced models (TCN) against autoregressive baselines.
- **Noise Mitigation:** Implement a cascading AI pipeline to filter non-price-sensitive information.
- **Temporal Integrity:** Enforce the "18h Rule" and Business Day adjustments at the database level.
- **Human-in-the-Loop Audit:** Allow researchers to inspect AI rationale for every triage decision.

---

## 4. Key Features

### 4.1. Cascading AI Inference Pipeline

- **Stage 1 (Triage):** `gpt-4o-mini` acts as a gatekeeper, assessing price sensitivity and relevance (Threshold ≥ 0.7).
- **Stage 2 (Deep Analysis):** `gpt-4o` extracts fine-grained sentiment scores, logits, and specific entities for relevant signals only.
- **Impact:** Reduces operational costs and prevents "regression to the mean" in the training dataset.

### 4.2. Backtest Research Lab (Dashboard)

- **Temporal Correlation Engine:** Interactive dual-axis chart (Recharts) mapping adjusted prices against aggregated sentiment.
- **Rigor Metrics Panel:** Real-time display of ROC-AUC, standard deviation, and Wilcoxon p-values for the TCN model.
- **Intelligence Feed:** Categorized news stream (Signals vs. Noise) with expanded "AI Reasoning" for transparency.

### 4.3. Methodological Controls

- **The 18h Rule:** Deterministic mapping of news published after market close to the next valid trading session.
- **Business Day Logic:** Automatic skipping of weekends and holidays for reference trading dates.
- **Unified Persistence:** Centralized SQLite/PostgreSQL schema (Prisma) ensuring referential integrity between news, sentiment, and market data.

---

## 5. Technical Stack

- **Framework:** Next.js 14 (Unified Architecture).
- **UI/UX:** Tailwind CSS + shadcn/ui (Fintech/Terminal Aesthetic).
- **Database:** Prisma ORM + SQLite (Dev) / PostgreSQL (Prod).
- **AI Engine:** Vercel AI SDK + OpenAI (GPT-4o).
- **Validation:** Playwright (E2E Testing) + Scipy/Statsmodels (Scientific Rigor).

---

## 6. Success Metrics (Research Framework)

- **Alpha Discovery:** Consistently beating the **Inertia Baseline (AUC 0.6058)**.
- **Data Density:** Percentage of articles classified as `isRelevant` (Target: < 20% of total corpus to maximize signal-to-noise ratio).
- **Statistical Significance:** Achieving a **p-value < 0.05** in Wilcoxon signed-rank tests across multi-seed runs.

---

## 7. Roadmap & Future Scope

- **Phase 5:** Integration of Institutional Feeds (Bloomberg/Reuters Terminal API).
- **Phase 6:** Expansion to B3 Small Caps (Lower market efficiency, higher potential for NLP alpha).
- **Phase 7:** Intraday Resolution (Moving from daily to 15-minute intervals).

---

_Document Version: 2.0.0_  
_Status: Production Ready / Methodologically Shielded_
