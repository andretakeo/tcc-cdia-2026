import { prisma } from "../lib/db";

async function checkDB() {
  const articles = await prisma.article.count();
  const sentiments = await prisma.sentimentAnalysis.count();
  const tickers = await prisma.ticker.findMany();
  const marketData = await prisma.marketData.count();

  console.log("=== Database Status ===");
  console.log(`Articles: ${articles}`);
  console.log(`Sentiments: ${sentiments}`);
  console.log(`Tickers: ${tickers.map(t => t.symbol).join(", ")}`);
  console.log(`Market Data Points: ${marketData}`);
}

checkDB();
