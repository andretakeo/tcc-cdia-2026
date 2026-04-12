import { NextResponse } from "next/server";
import { InfoMoneyScraper } from "@/lib/scraper/infomoney-scraper";
import { prisma } from "@/lib/db";

export async function POST(req: Request) {
  const ticker = "ITUB4";
  const source = "InfoMoney";
  const COOLDOWN_MS = 60 * 60 * 1000;

  try {
    const lastRun = await prisma.scrapeLog.findFirst({
      where: { tickerId: ticker, source: source, status: "SUCCESS" },
      orderBy: { executedAt: "desc" },
    });

    if (lastRun) {
      const timeSinceLastRun = Date.now() - lastRun.executedAt.getTime();
      if (timeSinceLastRun < COOLDOWN_MS) {
        return NextResponse.json({
          success: false,
          message: `Cooldown ativo.`,
        }, { status: 429 });
      }
    }

    const scraper = new InfoMoneyScraper();
    const stats = await scraper.run();

    await prisma.scrapeLog.create({
      data: { tickerId: ticker, source: source, status: "SUCCESS" }
    });

    return NextResponse.json({ success: true, stats });
  } catch (error) {
    return NextResponse.json({ error: "Fail" }, { status: 500 });
  }
}
