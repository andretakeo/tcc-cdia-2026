# Architecture Documentation - TCC CDIA 2026 (V2)

## Overview
The V2 architecture represents a shift from exploratory batch processing to a real-time, event-driven, and scientifically rigorous production environment. It is structured as a **Monorepo** managed by **Turborepo** and **pnpm**, ensuring modularity and scalability.

## Monorepo Structure
- **`apps/scraper`**: A serverless service (Next.js/Vercel Cron) responsible for continuous news ingestion.
- **`apps/inference`**: The intelligence layer. It implements a two-step cascading pipeline to process raw text into high-density financial signals.
- **`apps/dashboard`**: The visualization layer (Next.js) for monitoring real-time sentiment and asset correlations.
- **`packages/db`**: The shared data layer using **Prisma ORM** and **PostgreSQL**, maintaining strict relational integrity between news, sentiment, and market data.

---

## The Cascading Inference Pipeline
To solve the "noise drowning" problem identified in the experimental phase, we implemented a two-step validation process using the **Vercel AI SDK**:

### Step 1: Relevance Triage (The "Gatekeeper")
- **Model**: `gpt-4o-mini` (Fast & Cost-effective).
- **Goal**: Filter out market noise (procedural filings, non-impactful macro news).
- **Criteria**: Articles must be classified as `isPriceSensitive: true` and reach a `relevanceScore >= 0.7` to proceed.
- **Outcome**: Non-relevant articles are archived with a reasoning tag, preventing them from diluting the predictive model's dataset.

### Step 2: Deep Sentiment Analysis (The "Specialist")
- **Model**: `gpt-4o` (High reasoning capability).
- **Goal**: Extract domain-specific financial sentiment.
- **Data Extracted**:
    - **Sentiment Label**: POSITIVE, NEGATIVE, or NEUTRAL.
    - **Sentiment Score**: A continuous value from -1.0 to 1.0.
    - **Logits**: Raw probabilities for fine-grained analysis.
    - **Entities**: Identification of mentioned tickers, competitors, and key people.

---

## Data Flow & Event Orchestration
1.  **Ingestion**: `apps/scraper` fetches new articles from sources (InfoMoney, CVM, Reuters).
2.  **Persistence**: The raw article is saved to the PostgreSQL database.
3.  **Trigger**: The scraper sends a signed POST request to `apps/inference`.
4.  **Triage**: Inference service performs Step 1. If irrelevant, it stops here.
5.  **Refinement**: If relevant, it performs Step 2 and updates the database record.
6.  **Visualization**: The `apps/dashboard` reflects the new signal immediately via Server-Side Rendering or Webhooks.

---

## Technical Stack
- **Languages**: TypeScript (Strict mode), Python (Legacy research/Training).
- **Orchestration**: Turborepo.
- **Database**: Prisma + PostgreSQL (Neon/Supabase).
- **AI/LLM**: Vercel AI SDK + OpenAI.
- **Frontend**: Next.js 14, TailwindCSS, Shadcn/UI.

---
*This architecture is designed to capture Alpha by ensuring that only price-moving information enters the model training and prediction pipeline.*
