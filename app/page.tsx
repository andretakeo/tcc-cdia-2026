import { prisma } from "@/lib/db";
import { SentimentChart } from "@/components/SentimentChart";
import { RefreshNewsButton } from "@/components/RefreshNewsButton";

export default async function DashboardPage() {
  const marketData = await prisma.marketData.findMany({
    where: { tickerId: "ITUB4" },
    orderBy: { tradingDate: "asc" },
    take: 100,
  });

  const articles = await prisma.article.findMany({
    take: 10,
    orderBy: { publishedAt: "desc" },
    include: { sentiment: true, tickers: true },
  });

  const chartData = marketData.map(d => ({
    date: d.tradingDate.toLocaleDateString(),
    price: d.close,
    sentiment: d.logReturn || 0,
  }));

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <header className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Trading Signal Dashboard</h1>
            <p className="text-gray-500">Itaú Unibanco (ITUB4) • Unified Architecture</p>
          </div>
          <div className="flex gap-4 items-center">
            <RefreshNewsButton />
            <div className="bg-white px-4 py-2 rounded-lg border shadow-sm">
              <span className="text-sm font-medium text-gray-500">Status do Pipeline:</span>
              <span className="ml-2 text-green-600 font-bold">● Active</span>
            </div>
          </div>
        </header>

        <section className="mb-12">
          <h2 className="text-xl font-semibold mb-4">Correlação Preço vs. Sentimento</h2>
          <SentimentChart data={chartData} />
        </section>

        <section>
          <h2 className="text-xl font-semibold mb-4">Últimas Notícias Processadas</h2>
          <div className="grid gap-4">
            {articles.map((article) => (
              <div key={article.id} className="bg-white p-6 rounded-xl border shadow-sm transition-hover hover:border-blue-300">
                <div className="flex justify-between items-start mb-3">
                  <h3 className="text-lg font-bold text-gray-900 flex-1">{article.title}</h3>
                  <div className={`ml-4 px-3 py-1 rounded-full text-xs font-bold ${
                    article.sentiment?.isRelevant 
                      ? 'bg-green-100 text-green-700'
                      : 'bg-gray-100 text-gray-500'
                  }`}>
                    {article.sentiment?.isRelevant ? article.sentiment.sentimentLabel : 'DESCARTE'}
                  </div>
                </div>
                <p className="text-gray-600 text-sm mb-4 line-clamp-2 italic">
                  "{article.sentiment?.relevanceReason || 'Sem triagem'}"
                </p>
                <div className="flex items-center gap-6 text-xs text-gray-400 border-t pt-4">
                  <span className="font-medium text-gray-500">Fonte: {article.sourceName}</span>
                  <span>Pregão: {article.referenceTradingDate.toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
