import * as cheerio from "cheerio";
import { prisma } from "@/lib/db";
import { processAndSaveArticle } from "@/lib/inference/inference-service";

export interface ScrapedArticle {
  title: string;
  content: string;
  source: string;
  link: string;
  publishedAt: Date;
}

export class InfoMoneyScraper {
  private readonly rssUrl = "https://www.infomoney.com.br/mercados/feed/";

  public async run(): Promise<{ processed: number; errors: number }> {
    console.log(`[InfoMoneyScraper] Starting extraction...`);
    let processed = 0;
    let errors = 0;

    try {
      const latestLinks = await this.fetchLatestLinks();

      for (const link of latestLinks) {
        try {
          const existing = await prisma.article.findUnique({
            where: { sourceUrl: link },
            include: { sentiment: true }
          });

          if (existing && existing.sentiment) {
            console.log(`[InfoMoneyScraper] Already processed: ${link}`);
            continue;
          }

          if (!existing) {
            const scraped = await this.extractArticleContent(link);
            if (scraped) {
              // CHAMADA DIRETA À FUNÇÃO (Sem HTTP/Portas)
              await processAndSaveArticle({
                url: scraped.link,
                source: scraped.source,
                title: scraped.title,
                content: scraped.content,
                publishedAt: scraped.publishedAt,
                tickerSymbol: "ITUB4"
              });
              processed++;
            }
          } else {
            // Re-processar apenas inferência se o artigo já existir mas sem sentimento
            await processAndSaveArticle({
              url: existing.sourceUrl,
              source: existing.sourceName,
              title: existing.title,
              content: existing.content,
              publishedAt: existing.publishedAt,
              tickerSymbol: "ITUB4"
            });
            processed++;
          }
        } catch (err) {
          console.error(`[InfoMoneyScraper] Error processing ${link}:`, err);
          errors++;
        }
        await this.delay(1000);
      }
    } catch (error) {
      console.error(`[InfoMoneyScraper] Critical failure:`, error);
      throw error;
    }

    return { processed, errors };
  }

  private async fetchLatestLinks(): Promise<string[]> {
    const response = await fetch(this.rssUrl);
    const xml = await response.text();
    const links: string[] = [];
    const itemRegex = /<item>[\s\S]*?<link>(.*?)<\/link>[\s\S]*?<\/item>/g;
    let match;
    while ((match = itemRegex.exec(xml)) !== null) {
      links.push(match[1].trim());
    }
    return links.slice(0, 5);
  }

  private async extractArticleContent(url: string): Promise<ScrapedArticle | null> {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
    });

    if (!response.ok) return null;

    const html = await response.text();
    const $ = cheerio.load(html);

    const title = $("h1").first().text().trim();
    const timeTag = $("time").attr("datetime");
    const publishedAt = timeTag ? new Date(timeTag) : new Date();

    const paragraphs: string[] = [];
    const contentSelectors = ["article.im-article p", "div.article-content p", "div.single__content p", "article p"];

    for (const selector of contentSelectors) {
      $(selector).each((_, el) => {
        const text = $(el).text().trim();
        if (text.length > 20) paragraphs.push(text);
      });
      if (paragraphs.length > 0) break;
    }

    const content = paragraphs.join("\n\n").trim();
    if (!title || !content) return null;

    return { title, content, source: "InfoMoney", link: url, publishedAt };
  }

  private delay(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
