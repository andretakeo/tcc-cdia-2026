import { prisma } from "@/lib/db";
import { SentimentChart } from "@/components/SentimentChart";
import { RefreshNewsButton } from "@/components/RefreshNewsButton";
import { 
  TrendingUp, TrendingDown, Minus, Clock, Globe, ListFilter, 
  LayoutDashboard, History, BarChart3, AlertCircle, CheckCircle2,
  Beaker, Settings2, Search, Zap
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default async function ResearchLabPage({ 
  searchParams 
}: { 
  searchParams: { filter?: string } 
}) {
  const filter = searchParams.filter || "relevant";

  // 1. SimulaÃ§Ã£o de Dados de Backtesting (Refletindo a Fase 3.1)
  const marketData = await prisma.marketData.findMany({
    where: { tickerId: "ITUB4" },
    orderBy: { tradingDate: "asc" },
    take: 100,
  });

  const sentiments = await prisma.sentimentAnalysis.findMany({
    where: { isRelevant: true, tickerId: "ITUB4" },
    include: { article: true }
  });

  const sentimentMap: Record<string, number> = {};
  sentiments.forEach(s => {
    const date = s.article.referenceTradingDate.toISOString().split('T')[0];
    sentimentMap[date] = s.sentimentScore || 0;
  });

  const chartData = marketData.map(d => {
    const dateKey = d.tradingDate.toISOString().split('T')[0];
    return {
      date: d.tradingDate.toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit' }),
      price: d.close,
      sentiment: sentimentMap[dateKey] || 0,
    };
  });

  // NotÃ­cias filtradas
  const newsWhere: any = {};
  if (filter === "relevant") {
    newsWhere.sentiment = { isRelevant: true };
  } else if (filter === "noise") {
    newsWhere.sentiment = { isRelevant: false };
  }

  const newsFeed = await prisma.article.findMany({
    where: newsWhere,
    include: { sentiment: true },
    orderBy: { publishedAt: 'desc' },
    take: 6
  });

  return (
    <main className="flex-1 bg-[#f8fafc] min-h-screen">
      {/* Top Research Bar */}
      <div className="bg-white border-b border-slate-200 px-8 py-4 flex items-center justify-between sticky top-16 z-40">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-primary font-bold">
            <Beaker className="w-5 h-5" />
            <span className="tracking-tight uppercase text-xs">Backtest Lab v2.0</span>
          </div>
          <Separator orientation="vertical" className="h-6" />
          <div className="flex items-center gap-4">
            <div className="flex flex-col">
              <span className="text-[10px] font-black text-slate-400 uppercase">Horizonte</span>
              <select className="bg-transparent text-sm font-bold outline-none cursor-pointer">
                <option>21 Dias (Padrão)</option>
                <option>5 Dias (Curto)</option>
                <option>42 Dias (Longo)</option>
              </select>
            </div>
            <div className="flex flex-col border-l pl-4">
              <span className="text-[10px] font-black text-slate-400 uppercase">Modelo Ativo</span>
              <span className="text-sm font-bold text-blue-600">TCN + NLP Purified</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" className="gap-2 font-bold text-xs uppercase tracking-wider">
            <Settings2 className="w-3.5 h-3.5" /> Parâmetros
          </Button>
          <RefreshNewsButton />
        </div>
      </div>

      <div className="max-w-[1600px] mx-auto p-8 grid grid-cols-12 gap-8">
        
        {/* Left: Statistics & Hypotheses */}
        <div className="col-span-12 lg:col-span-3 space-y-6">
          <Card className="border-none shadow-sm bg-slate-900 text-white overflow-hidden">
            <div className="p-6">
              <h2 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-6">Métricas de Validação</h2>
              <div className="space-y-8">
                <div className="relative group">
                  <div className="absolute -left-2 top-0 bottom-0 w-1 bg-blue-500 rounded-full" />
                  <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1 pl-2">TCN ROC-AUC</p>
                  <p className="text-5xl font-mono tracking-tighter pl-2">0.562<span className="text-xs text-slate-600 font-sans ml-1 tracking-normal">1</span></p>
                </div>
                <div className="space-y-4 pt-4 border-t border-slate-800">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-500 font-bold uppercase tracking-tighter">Baseline Inércia</span>
                    <span className="font-mono text-slate-300">0.6058</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-500 font-bold uppercase tracking-tighter">Wilcoxon p-value</span>
                    <Badge variant="outline" className="text-[10px] font-mono border-slate-700 text-orange-400">0.4821</Badge>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-blue-600 p-4 text-center">
              <p className="text-[10px] font-black uppercase tracking-widest">Hipótese Rejeitada</p>
            </div>
          </Card>

          <Card className="border-slate-200 shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-xs font-black uppercase tracking-widest text-slate-400 flex items-center gap-2">
                <Zap className="w-3.5 h-3.5 text-orange-500" /> Insights de Pesquisa
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-xs leading-relaxed text-slate-600 font-medium">
                O sinal de sentimento de portais como o <span className="text-slate-900 font-bold">InfoMoney</span> apresenta alta eficiência de mercado. O preço absorve a notícia em menos de 18h.
              </p>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <p className="text-[10px] text-slate-400 font-bold uppercase mb-2">Sugestão de Próximo Passo</p>
                <p className="text-[11px] text-slate-700 leading-snug italic font-serif">
                  "Testar a arquitetura em ativos de baixa liquidez (Small Caps) para capturar ineficiências maiores."
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Center: Interactive Visualizer */}
        <div className="col-span-12 lg:col-span-9 space-y-8">
          <Card className="border-none shadow-md overflow-hidden">
            <CardHeader className="bg-white border-b pb-4 flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-lg font-black text-slate-900">Análise de Correlação Temporal</CardTitle>
                <CardDescription className="text-xs font-medium">Preço Ajustado vs. Sentimento Agregado (18h Rule)</CardDescription>
              </div>
              <div className="flex gap-2">
                <Badge variant="outline" className="text-[10px] font-bold">LOG_RETURNS: ON</Badge>
                <Badge variant="outline" className="text-[10px] font-bold">SENT_FILTER: 0.70</Badge>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="p-6">
                <SentimentChart data={chartData} />
              </div>
              <div className="bg-slate-50 border-t p-4 flex items-center justify-center gap-8">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500" />
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">ITUB4 Price</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-emerald-400" />
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">AI Sentiment (Relevant Only)</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Intelligent News Feed */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-black text-slate-900 uppercase tracking-widest flex items-center gap-2">
                <Search className="w-4 h-4 text-primary" /> Auditoria de Sinais em Tempo Real
              </h2>
              <div className="flex bg-white border rounded-lg p-1 gap-1">
                <Link href="/?filter=relevant">
                  <Button variant="ghost" size="sm" className={`h-7 text-[10px] font-bold uppercase px-3 ${filter === 'relevant' ? 'bg-slate-100' : ''}`}>Sinais (Relevant)</Button>
                </Link>
                <Link href="/?filter=all">
                  <Button variant="ghost" size="sm" className={`h-7 text-[10px] font-bold uppercase px-3 ${filter === 'all' ? 'bg-slate-100' : ''}`}>Tudo</Button>
                </Link>
                <Link href="/?filter=noise">
                  <Button variant="ghost" size="sm" className={`h-7 text-[10px] font-bold uppercase px-3 ${filter === 'noise' ? 'bg-slate-100' : ''}`}>RuÃ­do</Button>
                </Link>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {newsFeed.map((article) => (
                <Link href={`/article/${article.id}`} key={article.id}>
                  <Card className="h-full hover:border-blue-400/50 transition-all group cursor-pointer border-slate-200">
                    <CardHeader className="p-5 pb-2">
                      <div className="flex justify-between items-start mb-2 gap-4">
                        <Badge variant="secondary" className="bg-blue-50 text-blue-700 hover:bg-blue-100 border-none text-[9px] font-black uppercase tracking-tighter italic">
                          {article.sourceName}
                        </Badge>
                        {article.sentiment?.isRelevant ? (
                          article.sentiment.sentimentClass === 'POSITIVE' ? 
                            <TrendingUp className="w-4 h-4 text-emerald-500" /> : <TrendingDown className="w-4 h-4 text-red-500" />
                        ) : <Minus className="w-4 h-4 text-slate-300" />}
                      </div>
                      <CardTitle className="text-sm font-bold leading-snug line-clamp-2 group-hover:text-blue-600 transition-colors tracking-tight">
                        {article.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="px-5 pb-5 pt-0">
                      <div className="bg-slate-50 border border-slate-100 rounded-xl p-3 mb-3">
                        <p className="text-[11px] text-slate-500 font-serif leading-relaxed line-clamp-2 italic">
                          "{article.sentiment?.relevanceReason || 'Aguardando processamento...'}"
                        </p>
                      </div>
                      <div className="flex items-center justify-between text-[9px] font-black text-slate-400 uppercase tracking-widest">
                        <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {article.referenceTradingDate.toLocaleDateString('pt-BR')}</span>
                        <span className="text-blue-500">Details →</span>
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
        </div>

      </div>
    </main>
  );
}
