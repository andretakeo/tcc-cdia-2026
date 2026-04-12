import fs from "fs";
import { prisma } from "@tcc/db";

/**
 * Script de Exportação: Transforma os dados triados do Postgres em um
 * dataset tabular pronto para treinamento em Python.
 * 
 * Regras aplicadas:
 * 1. Apenas notícias marcadas como isRelevant = true.
 * 2. Agrupamento por referenceTradingDate.
 * 3. Merge com MarketData (Preços).
 */
async function exportPurifiedDataset() {
  console.log("[ETL] Iniciando extração de dados purificados...");

  // 1. Buscar MarketData (Preços)
  const marketData = await prisma.marketData.findMany({
    where: { tickerId: "ITUB4" },
    orderBy: { tradingDate: "asc" },
  });

  // 2. Buscar SentimentAnalysis Relevante
  const relevantSentiment = await prisma.sentimentAnalysis.findMany({
    where: { 
      tickerId: "ITUB4",
      isRelevant: true 
    },
    include: {
      article: true
    }
  });

  console.log(`[ETL] Encontrados ${marketData.length} dias de mercado e ${relevantSentiment.length} notícias relevantes.`);

  // 3. Agrupar sentimento por data de negociação
  const dailySentiment: Record<string, any> = {};

  relevantSentiment.forEach((s) => {
    const dateStr = s.article.referenceTradingDate.toISOString().split("T")[0];
    if (!dailySentiment[dateStr]) {
      dailySentiment[dateStr] = {
        count: 0,
        sumScore: 0,
        sumPos: 0,
        sumNeg: 0,
        sumNeu: 0,
      };
    }
    dailySentiment[dateStr].count++;
    dailySentiment[dateStr].sumScore += s.sentimentScore || 0;
    dailySentiment[dateStr].sumPos += s.logitPos || 0;
    dailySentiment[dateStr].sumNeg += s.logitNeg || 0;
    dailySentiment[dateStr].sumNeu += s.logitNeu || 0;
  });

  // 4. Montar o Dataset Final (Join)
  const finalRows = marketData.map((m) => {
    const dateStr = m.tradingDate.toISOString().split("T")[0];
    const s = dailySentiment[dateStr] || { count: 0, sumScore: 0, sumPos: 0, sumNeg: 0, sumNeu: 0 };
    
    return {
      date: dateStr,
      close: m.close,
      volume: m.volume.toString(),
      returns: m.logReturn,
      volatility: m.volatility,
      // Features de Sentimento (Médias Diárias)
      news_count: s.count,
      avg_sentiment: s.count > 0 ? s.sumScore / s.count : 0,
      avg_logit_pos: s.count > 0 ? s.sumPos / s.count : 0,
      avg_logit_neg: s.count > 0 ? s.sumNeg / s.count : 0,
      avg_logit_neu: s.count > 0 ? s.sumNeu / s.count : 0,
    };
  });

  const csvHeader = Object.keys(finalRows[0]).join(",");
  const csvRows = finalRows.map(row => Object.values(row).join(",")).join("\n");
  const csvContent = `${csvHeader}\n${csvRows}`;

  const outputPath = "../../V2/3.model-expansion/purified_itub4_dataset.csv";
  fs.writeFileSync(outputPath, csvContent);

  console.log(`[ETL] Dataset purificado exportado para: ${outputPath}`);
}

exportPurifiedDataset();
