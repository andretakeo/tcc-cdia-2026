import { InfoMoneyScraper } from "../lib/scraper/infomoney-scraper";

async function main() {
  const scraper = new InfoMoneyScraper();
  const stats = await scraper.run();
  console.log("Stats:", stats);
}

main();
