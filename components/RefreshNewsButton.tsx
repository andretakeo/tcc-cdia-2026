"use client";

import { useState } from "react";
import { RefreshCcw, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

export function RefreshNewsButton() {
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [message, setMessage] = useState<string | null>(null);

  async function handleRefresh() {
    setStatus("loading");
    setMessage(null);
    try {
      const response = await fetch("/api/scrape", { method: "POST" });
      const data = await response.json();
      
      if (response.ok) {
        setStatus("success");
        setTimeout(() => window.location.reload(), 1500);
      } else {
        setStatus("error");
        setMessage(data.message || "Erro.");
      }
    } catch (error) {
      setStatus("error");
      setMessage("Conexão falhou.");
    }
  }

  return (
    <div className="relative">
      <Button
        onClick={handleRefresh}
        disabled={status === "loading"}
        variant={status === "success" ? "secondary" : "default"}
        className="gap-2"
      >
        {status === "loading" ? (
          <RefreshCcw className="w-4 h-4 animate-spin" />
        ) : status === "success" ? (
          <CheckCircle2 className="w-4 h-4 text-emerald-500" />
        ) : (
          <RefreshCcw className="w-4 h-4" />
        )}
        {status === "loading" ? "Triando Notícias..." : "Atualizar Sinal"}
      </Button>
      
      {message && (
        <div className="absolute top-full mt-2 right-0 bg-destructive/10 text-destructive text-[10px] px-2 py-1 rounded border border-destructive/20 whitespace-nowrap flex items-center gap-1">
          <AlertCircle className="w-3 h-3" /> {message}
        </div>
      )}
    </div>
  );
}
