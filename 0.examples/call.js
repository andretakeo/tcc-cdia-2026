/**
 * Exploração da API REST do InfoMoney (WordPress /wp-json/wp/v2/posts)
 * ─────────────────────────────────────────────────────────────────────
 * O InfoMoney utiliza o WordPress como CMS, expondo seus artigos via a
 * REST API padrão do WP no endpoint `/wp-json/wp/v2/posts`.
 *
 * Por que o InfoMoney foi escolhido como fonte de dados:
 *   - Cobertura ampla de ações brasileiras (B3)
 *   - API pública sem necessidade de autenticação ou chave de API
 *   - Campos estruturados ricos: id, title, content, date, categories, tags
 *   - Suporte a paginação e filtros temporais (modified_after)
 *
 * Parâmetros principais da chamada:
 *   - per_page   : número de posts por página (máx. 100)
 *   - page       : página atual para paginação
 *   - search     : termo de busca (ticker da ação, ex: "WEGE3")
 *   - orderby    : campo de ordenação ("modified" para ordem cronológica)
 *   - order      : direção da ordenação ("desc" = mais recente primeiro)
 *   - status     : filtra apenas posts publicados ("publish")
 *   - modified_after : filtro temporal ISO 8601 para limitar o período
 *
 * Campos relevantes do response JSON:
 *   - id         : identificador único do post
 *   - title      : objeto com campo "rendered" contendo o título em HTML
 *   - content    : objeto com campo "rendered" contendo o corpo completo em HTML
 *   - date       : data de publicação no formato ISO 8601
 *   - categories : array de IDs numéricos das categorias do post
 *   - tags       : array de IDs numéricos das tags do post
 *
 * Exemplo de chamada HTTP abaixo — busca posts sobre WEGE3 modificados
 * após 2025-03-15, ordenados do mais recente ao mais antigo:
 */
fetch(
  "https://www.infomoney.com.br/wp-json/wp/v2/posts?_embed=wp:featuredmedia&per_page=25&page=1&search=WEGE3&orderby=modified&order=desc&status=publish&context=view&modified_after=2025-03-15T21%3A20%3A35.676Z",
  {
    headers: {
      accept: "application/json",
      "content-type": "application/json",
      "sec-ch-ua":
        '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
      "sec-ch-ua-mobile": "?0",
      "sec-ch-ua-platform": '"macOS"',
      Referer: "https://www.infomoney.com.br/busca/?q=PETR4",
    },
    body: null,
    method: "GET",
  },
);
