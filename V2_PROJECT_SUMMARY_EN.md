# Technical Report: Evolution of the Stock Prediction Pipeline (V2)

## Executive Summary
This document details the transition of the "Stock Price Direction Prediction with NLP and News Sentiment" project from its exploratory phase (V1) to a scientifically rigorous and production-ready ecosystem (V2). The evolution focused on two pillars: **Methodological Shielding** to eliminate evaluation biases and **Architectural Modernization** to enable real-time scalability.

---

## Phase 1: Methodological Shielding (Scientific Rigor)
The primary goal was to ensure that model results reflect performance in a realistic trading environment.

### 1.1 Expanding-Window Cross-Validation
*   **The Problem:** The initial static split (70% train / 15% val / 15% test) in non-stationary financial series generated illusory results and overfitted to specific market regimes.
*   **The Solution:** Implemented a **Walk-Forward** evaluation loop. The model trains on months $1$ to $t$ and tests on month $t+1$, expanding the window iteratively.
*   **Outcome:** Eliminated "look-ahead" leakage and provided a statistically significant distribution of performance across time.

### 1.2 Look-Ahead Bias Correction (The 18h Rule)
*   **The Problem:** Standard daily merges often associate evening news (published after market close) with the same day's closing price, leaking "future" information into the model.
*   **The Solution:** Implemented a strict market-closure rule. News published after **18:00:00 (B3 Close)** are automatically shifted to the next business day's features.
*   **Outcome:** Validated that sentiment only influences the *next available* trading session.

### 1.3 Naive Baselines Establishment
*   **The Problem:** High accuracy numbers (e.g., 76%) are meaningless without a baseline for unbalanced classes.
*   **The Solution:** Created three benchmark models:
    1.  **Majority:** Always predicts "Up". (Hit **61.8% Accuracy**).
    2.  **Inertia:** Predicts that the next 21 days will follow the direction of the last 21 days. (Hit **60.6% Accuracy**).
    3.  **Probabilistic:** Randomly predicts based on class distribution.
*   **Discovery:** The "0.709 AUC" from the V1 Transformer was identified as a sampling artifact, as simple heuristics already achieve ~62% accuracy.

---

## Phase 2: Feature Engineering & Isolation Experiments
Focus on isolating variables to understand the true impact of textual features.

### 2.1 Dimensionality Ablation Test
*   **Experiment:** Compared **FinBERT (5 dimensions)** against **Ollama Embeddings compressed via PCA (5 dimensions)**.
*   **Result:** Ollama PCA (AUC 0.4978) outperformed FinBERT (AUC 0.4244).
*   **Finding:** The performance gains in V1 were driven by **dimensionality reduction** (noise filtering) rather than the specific financial knowledge of the language model.

### 2.2 Horizon Sweep
*   **Experiment:** Iterated over multiple prediction horizons: $h \in \{1, 2, 5, 10, 21, 42\}$ business days.
*   **Finding:** Sentiment signal from retail portals (InfoMoney) is extremely weak for short-term prediction and only shows a slight recovery at **42 days (AUC 0.5174)**.

---

## Phase 3: Model Expansion & Multi-Source Integration
Validating advanced architectures and expanding the data footprint.

### 3.1 TCN (Temporal Convolutional Network) Scrutiny
*   **Result:** Re-evaluated the TCN architecture. While more stable than the Transformer (less variance across seeds), it achieved an **AUC of 0.56**, still struggling to consistently beat the autoregressive price baseline.

### 3.2 Multi-Source News Ingestion
*   **Experiment:** Integrated data from **Reuters, Google News, and CVM (Official Filings)**.
*   **Result:** Performance dropped to **AUC 0.39**.
*   **Finding:** Without a rigorous relevance filter, adding more sources introduced excessive noise, confusing the model with non-price-sensitive information.

---

## Phase 4: Production Architecture (Modernization)
Transitioned from exploratory notebooks to a robust, event-driven monorepo.

### 4.1 Monorepo Structure (Turborepo + pnpm)
*   **`packages/db`**: Shared data layer using **Prisma ORM** and **PostgreSQL**. Models `Tickers`, `Articles`, and `SentimentAnalysis` with strict referential integrity.
*   **`apps/scraper`**: Automated ingestion service via **Vercel Cron Jobs**, designed to fetch news periodically.
*   **`apps/inference`**: Real-time processing API using the **Vercel AI SDK**. It transforms raw text into structured JSON (sentiment + entities) using Zod schemas.
*   **`apps/dashboard`**: **Next.js** interface for real-time monitoring of sentiment trends correlated with price movements.

### 4.2 Real-Time AI Pipeline
Instead of batch processing, every new article now triggers an event:
1.  **Ingest:** Scraper finds a new article.
2.  **Store:** Article is saved to PostgreSQL.
3.  **Analyze:** Inference service extracts sentiment and entities via LLM.
4.  **Visualize:** Dashboard updates immediately for the end-user.

---

## Final Conclusion
The V2 evolution successfully debunked the "mirage" of high predictive power in retail news sentiment by applying rigorous scientific controls. The project now stands as a **production-ready platform** that prioritizes data quality and statistical validity over raw, unverified metrics, providing a solid foundation for future research into institutional news and high-frequency data.
