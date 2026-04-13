import { NextResponse } from "next/server";
import { InfoMoneyScraper } from "@/lib/scraper/infomoney-scraper";
import { prisma } from "@/lib/db";

export async function POST(req: Request) {
  const ticker = "ITUB4";
  const source = "InfoMoney";

  try {
    // Cooldown removido para permitir execuções sob demanda sem restrição de tempo.
    
    const scraper = new InfoMoneyScraper();
    const stats = await scraper.run();

    // Registrar execução apenas para histórico
    await prisma.scrapeLog.create({
      data: { 
        tickerId: ticker, 
        source: source, 
        status: stats.errors === 0 ? "SUCCESS" : "PARTIAL_ERROR" 
      }
    });

    return NextResponse.json({ success: true, stats });
  } catch (error) {
    console.error("Scraper Route Error:", error);
    return NextResponse.json({ error: "Fail" }, { status: 500 });
  }
}
