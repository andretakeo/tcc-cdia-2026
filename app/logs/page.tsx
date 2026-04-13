import { prisma } from "@/lib/db";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Clock, CheckCircle2, XCircle } from "lucide-react";

export default async function LogsPage() {
  const logs = await prisma.scrapeLog.findMany({
    orderBy: { executedAt: "desc" },
    take: 50,
  });

  return (
    <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-8">
      <div className="flex items-center gap-3 mb-8">
        <Clock className="w-8 h-8 text-primary" />
        <h1 className="text-3xl font-black tracking-tight text-slate-900">Histórico de Execuções</h1>
      </div>

      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-slate-50 text-slate-500 font-bold uppercase text-[10px] tracking-widest border-b">
                <tr>
                  <th className="px-6 py-4">Data/Hora</th>
                  <th className="px-6 py-4">Ativo</th>
                  <th className="px-6 py-4">Fonte</th>
                  <th className="px-6 py-4">Estado</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {logs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-50/50 transition-colors">
                    <td className="px-6 py-4 font-mono text-slate-500">
                      {log.executedAt.toLocaleString('pt-BR')}
                    </td>
                    <td className="px-6 py-4 font-bold text-slate-700">
                      {log.tickerId}
                    </td>
                    <td className="px-6 py-4 text-slate-600">
                      {log.source}
                    </td>
                    <td className="px-6 py-4">
                      {log.status === "SUCCESS" ? (
                        <div className="flex items-center gap-2 text-emerald-600 font-medium">
                          <CheckCircle2 className="w-4 h-4" /> Sucesso
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 text-red-600 font-medium">
                          <XCircle className="w-4 h-4" /> Erro
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
                {logs.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-6 py-12 text-center text-slate-400 italic">
                      Nenhum log de execução encontrado.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}
