import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "TCC Terminal | Stock Sentiment V2",
  description: "NLP-driven financial research platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR">
      <body className={`${inter.className} antialiased`}>
        <div className="min-h-screen flex flex-col">
          <nav className="border-b bg-white/80 backdrop-blur-md sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-xl">
                  T
                </div>
                <span className="font-bold text-lg tracking-tight">QUANT-NLP <span className="text-blue-600">V2</span></span>
              </div>
              <div className="text-xs text-slate-400 font-mono hidden md:block">
                ENVIRONMENT: PRODUCTION_SIM | DATA_RIGOR: STRICT
              </div>
            </div>
          </nav>
          {children}
        </div>
      </body>
    </html>
  );
}
