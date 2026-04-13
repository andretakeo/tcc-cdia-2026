import { generateObject } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import { prisma } from "@/lib/db";

// ==========================================
// 1. Lógica Científica: A Regra das 18h
// ==========================================

/**
 * Calcula o dia útil de referência para o impacto de uma notícia no mercado (B3).
 * Se publicada após as 18h00 (BRT), empurra para o dia útil seguinte.
 * Ignora fins de semana (Sábado -> Segunda, Domingo -> Segunda).
 */
export function calculateReferenceTradingDate(publishedAt: Date): Date {
  const date = new Date(publishedAt);

  // Conversão simples para o fuso horário de Brasília (UTC-3)
  const brtHour = (date.getUTCHours() - 3 + 24) % 24;

  // Regra das 18h: Se for 18h ou mais, o impacto passa para o dia seguinte
  if (brtHour >= 18) {
    date.setUTCDate(date.getUTCDate() + 1);
  }

  // Ajuste de Fins de Semana (Business Days)
  let dayOfWeek = date.getUTCDay();
  if (dayOfWeek === 6) { // Sábado
    date.setUTCDate(date.getUTCDate() + 2);
  } else if (dayOfWeek === 0) { // Domingo
    date.setUTCDate(date.getUTCDate() + 1);
  }

  // Recalcular após o ajuste de data para garantir que não caímos em novo fim de semana
  dayOfWeek = date.getUTCDay();
  if (dayOfWeek === 6) date.setUTCDate(date.getUTCDate() + 2);
  if (dayOfWeek === 0) date.setUTCDate(date.getUTCDate() + 1);

  date.setUTCHours(0, 0, 0, 0);
  return date;
}

// ==========================================
// 2. Schemas de Zod para IA
// ==========================================

const ArticleTriageSchema = z.object({
  isPriceSensitive: z.boolean(),
  relevanceScore: z.number().min(0).max(1),
  reasoning: z.string()
});

const SentimentSchema = z.object({
  sentimentClass: z.enum(['POSITIVE', 'NEGATIVE', 'NEUTRAL']),
  sentimentScore: z.number().min(-1).max(1),
  entities: z.array(z.string()),
  keyTakeaway: z.string()
});

// ==========================================
// 3. Orquestrador do Pipeline
// ==========================================

export async function processAndSaveArticle(articleData: {
  url: string;
  source: string;
  title: string;
  content: string;
  publishedAt: Date;
  tickerSymbol: string;
}) {
  console.log(`[Pipeline] Processing ${articleData.tickerSymbol}: "${articleData.title}"`);

  try {
    const referenceTradingDate = calculateReferenceTradingDate(articleData.publishedAt);

    // 1. Upsert Article
    const article = await prisma.article.upsert({
      where: { sourceUrl: articleData.url },
      update: {},
      create: {
        sourceUrl: articleData.url,
        sourceName: articleData.source,
        title: articleData.title,
        content: articleData.content,
        publishedAt: articleData.publishedAt,
        referenceTradingDate: referenceTradingDate,
        tickers: {
          connectOrCreate: {
            where: { symbol: articleData.tickerSymbol },
            create: { symbol: articleData.tickerSymbol, companyName: articleData.tickerSymbol }
          }
        }
      }
    });

    const existingSentiment = await prisma.sentimentAnalysis.findFirst({
      where: { articleId: article.id, tickerId: articleData.tickerSymbol }
    });

    if (existingSentiment) {
      return existingSentiment;
    }

    // --- MOCK MODE IF NO API KEY ---
    if (!process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY === "your-key-here") {
      console.log("[Pipeline] ⚠️ No API Key found. Running in MOCK MODE.");       
      const isRelevant = Math.random() > 0.3; // 70% chance of being relevant        

      return await prisma.sentimentAnalysis.create({
        data: {
          articleId: article.id,
          tickerId: articleData.tickerSymbol,
          isRelevant: isRelevant,
          relevanceScore: isRelevant ? 0.85 : 0.4,
          relevanceReason: "MOCK: Automated triage reasoning simulation.",
          sentimentClass: isRelevant ? (Math.random() > 0.5 ? "POSITIVE" : "NEGATIVE") : "NEUTRAL",
          sentimentScore: isRelevant ? (Math.random() * 2 - 1) : 0,
          entities: JSON.stringify(["MOCK_ENTITY"]),
          keyTakeaway: "MOCK: Key takeaway simulation.",
        }
      });
    }

    // 2. STEP 1: Triage (gpt-4o-mini)
    const { object: triage } = await generateObject({
      model: openai('gpt-4o-mini'),
      schema: ArticleTriageSchema,
      system: `You are a quantitative analyst. Filter noise. Assess if news changes fundamentals for ${articleData.tickerSymbol}.`,
      prompt: `Title: ${articleData.title}\nContent: ${articleData.content.substring(0, 1500)}`
    });

    const isRelevant = triage.isPriceSensitive && triage.relevanceScore >= 0.7;      

    if (!isRelevant) {
      return await prisma.sentimentAnalysis.create({
        data: {
          articleId: article.id,
          tickerId: articleData.tickerSymbol,
          isRelevant: false,
          relevanceScore: triage.relevanceScore,
          relevanceReason: triage.reasoning,
        }
      });
    }

    // 3. STEP 2: Deep Sentiment (gpt-4o)
    console.log(`[Pipeline] ✅ Relevant! Extracting deep sentiment...`);
    const { object: sentiment } = await generateObject({
      model: openai('gpt-4o'),
      schema: SentimentSchema,
      system: `Analyze price impact sentiment for ${articleData.tickerSymbol}.`,     
      prompt: `Title: ${article.title}\nContent: ${article.content}`
    });

    return await prisma.sentimentAnalysis.create({
      data: {
        articleId: article.id,
        tickerId: articleData.tickerSymbol,
        isRelevant: true,
        relevanceScore: triage.relevanceScore,
        relevanceReason: triage.reasoning,
        sentimentClass: sentiment.sentimentClass,
        sentimentScore: sentiment.sentimentScore,
        entities: JSON.stringify(sentiment.entities),
        keyTakeaway: sentiment.keyTakeaway,
      }
    });

  } catch (error) {
    console.error(`[Pipeline Error] ${articleData.url}:`, error);
    throw error;
  }
}
