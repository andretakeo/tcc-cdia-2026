import { prisma } from "@/lib/db";
import { notFound } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ChevronLeft, Brain, BarChart3, Newspaper } from "lucide-react";
import Link from "next/link";

export default async function ArticleDetailPage({ params }: { params: { id: string } }) {
  const article = await prisma.article.findUnique({
    where: { id: params.id },
    include: { sentiment: true, tickers: true },
  });

  if (!article) notFound();

  const sentiment = article.sentiment;

  return (
    <main className="flex-1 max-w-4xl mx-auto w-full px-4 py-8">
      <Link href="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-primary mb-6 transition-colors font-medium">
        <ChevronLeft className="w-4 h-4" /> Voltar ao Dashboard
      </Link>

      <div className="space-y-8">
        {/* News Section */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-primary font-bold uppercase text-[10px] tracking-widest">
            <Newspaper className="w-4 h-4" /> Conteúdo do Artigo
          </div>
          <h1 className="text-4xl font-black text-slate-900 leading-[1.1] tracking-tight">{article.title}</h1>
          <div className="flex items-center gap-4 text-xs text-muted-foreground font-medium">
            <span>Fonte: {article.sourceName}</span>
            <span>Publicado: {article.publishedAt.toLocaleString('pt-BR')}</span>
            <span>Pregão Alvo: {article.referenceTradingDate.toLocaleDateString('pt-BR')}</span>
          </div>
          <div className="prose prose-slate max-w-none text-slate-600 leading-relaxed text-lg bg-white p-8 rounded-3xl border shadow-sm">
            {article.content.split('\n\n').map((p, i) => (
              <p key={i} className="mb-4">{p}</p>
            ))}
          </div>
        </section>

        {/* AI Analysis Section */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-primary font-bold uppercase text-[10px] tracking-widest">
            <Brain className="w-4 h-4" /> Auditoria de IA (V2 Rigor)
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className={sentiment?.isRelevant ? "border-emerald-100 bg-emerald-50/20" : "border-orange-100 bg-orange-50/20"}>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs uppercase tracking-widest text-muted-foreground">Triagem</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-black mb-1">{sentiment?.isRelevant ? "RELEVANTE" : "RUÍDO"}</div>
                <p className="text-xs text-muted-foreground leading-snug italic">"{sentiment?.relevanceReason}"</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs uppercase tracking-widest text-muted-foreground">Sentimento</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant={sentiment?.sentimentClass === 'POSITIVE' ? 'success' : sentiment?.sentimentClass === 'NEGATIVE' ? 'destructive' : 'secondary'} className="text-sm font-black">
                    {sentiment?.sentimentClass || "N/A"}
                  </Badge>
                </div>
                <div className="text-xs font-mono text-muted-foreground tracking-tighter">SCORE: {sentiment?.sentimentScore?.toFixed(4) || "0.0000"}</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs uppercase tracking-widest text-muted-foreground">Entidades</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  {sentiment?.entities ? JSON.parse(sentiment.entities as string).map((e: string) => (
                    <Badge key={e} variant="outline" className="bg-white text-[10px] uppercase font-bold">{e}</Badge>
                  )) : "Nenhuma"}
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </main>
  );
}
