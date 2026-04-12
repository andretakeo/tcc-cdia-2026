"use client";

import { useState } from "react";
import { RefreshCcw } from "lucide-react";

export function RefreshNewsButton() {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function handleRefresh() {
    setLoading(true);
    setMessage(null);
    try {
      const response = await fetch("http://localhost:3002/api/scrape", {
        method: "POST",
      });
      const data = await response.json();
      
      if (response.ok) {
        setMessage(`Sucesso! ${data.stats.processed} notícias novas.`);
        window.location.reload(); // Recarregar para ver as novidades
      } else {
        setMessage(data.message || "Erro ao atualizar.");
      }
    } catch (error) {
      setMessage("Erro de conexão com o scraper.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col items-end gap-2">
      <button
        onClick={handleRefresh}
        disabled={loading}
        className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
      >
        <RefreshCcw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        {loading ? "Atualizando..." : "Atualizar Notícias"}
      </button>
      {message && (
        <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
          {message}
        </span>
      )}
    </div>
  );
}
