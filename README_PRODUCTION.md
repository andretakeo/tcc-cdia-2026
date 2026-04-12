# TCC CDIA 2026 - Modernização da Arquitetura (Produção)

Este repositório foi migrado para uma estrutura de monorepo utilizando **Turborepo** e **pnpm**, focada em escalabilidade, rigor de dados e processamento em tempo real via IA.

## Estrutura do Projeto

```
.
├── apps/
│   ├── dashboard/    # Interface Next.js para visualização de dados e predições.
│   ├── inference/    # API Next.js que processa notícias via Vercel AI SDK.
│   └── scraper/      # Serviço de coleta de notícias via Vercel Cron Jobs.
├── packages/
│   └── db/           # Pacote compartilhado com Prisma ORM e Schema PostgreSQL.
├── V1/               # Histórico da fase exploratória (Notebooks).
└── V2/               # Experimentos de rigor científico e ablação.
```

## Tecnologias Principais

- **Turborepo**: Orquestração de builds e cache.
- **pnpm**: Gerenciamento eficiente de dependências em monorepo.
- **Next.js**: Framework para aplicações Web e APIs Serverless.
- **Prisma**: ORM para persistência rigorosa em PostgreSQL.
- **Vercel AI SDK**: Extração estruturada de sentimentos e entidades via LLMs.
- **TailwindCSS**: Estilização da interface.

## Como Começar

1.  Instale as dependências:
    ```bash
    pnpm install
    ```

2.  Configure as variáveis de ambiente:
    Crie um arquivo `.env` na raiz com:
    ```
    DATABASE_URL="postgresql://..."
    OPENAI_API_KEY="sk-..."
    ```

3.  Gere o cliente do banco de dados:
    ```bash
    pnpm --filter @tcc/db generate
    ```

4.  Inicie o ambiente de desenvolvimento:
    ```bash
    pnpm dev
    ```

## Pipeline de Dados em Tempo Real

1.  **Scraper (apps/scraper)**: Coleta notícias periodicamente via Cron.
2.  **Database (packages/db)**: Salva o artigo bruto e aciona a inferência.
3.  **Inference (apps/inference)**: Extrai sentimento e entidades (Zod schema) usando o AI SDK.
4.  **Dashboard (apps/dashboard)**: Exibe os resultados correlacionados com preços de mercado.
