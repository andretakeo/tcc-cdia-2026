import { prisma } from "../lib/db";

async function seedMarketData() {
  const ticker = await prisma.ticker.upsert({
    where: { symbol: "ITUB4" },
    update: {},
    create: { symbol: "ITUB4", companyName: "Itaú Unibanco" },
  });

  const data = [];
  const now = new Date();
  for (let i = 0; i < 30; i++) {
    const date = new Date();
    date.setDate(now.getDate() - i);
    date.setUTCHours(0, 0, 0, 0);
    
    data.push({
      tickerId: ticker.symbol,
      tradingDate: date,
      open: 30 + Math.random() * 5,
      high: 35 + Math.random() * 5,
      low: 25 + Math.random() * 5,
      close: 30 + Math.random() * 5,
      volume: BigInt(Math.floor(Math.random() * 1000000)),
    });
  }

  for (const d of data) {
    await prisma.marketData.upsert({
      where: {
        tickerId_tradingDate: {
          tickerId: d.tickerId,
          tradingDate: d.tradingDate,
        },
      },
      update: {},
      create: d,
    });
  }

  console.log("Market data seeded!");
}

seedMarketData();
