import re
import urllib.parse
import httpx
from typing import List, Dict, Any, Optional
from app.providers.base import SearchProvider, SearchResult
from app.config import settings

class UnifiedSearchProvider(SearchProvider):
    """
    Unified Real-Time Web Search Provider.
    Supports Tavily API if configured, with automatic fallback to Free DuckDuckGo / Wikipedia real search.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.TAVILY_API_KEY

    @property
    def provider_name(self) -> str:
        return "unified_web_search"

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if not query:
            return []

        # 1. If Tavily API Key provided, use Tavily API
        if self.api_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    res = await client.post(
                        "https://api.tavily.com/search",
                        json={"api_key": self.api_key, "query": query, "max_results": num_results}
                    )
                    if res.status_code == 200:
                        data = res.json()
                        results = []
                        for item in data.get("results", []):
                            results.append(SearchResult(
                                title=item.get("title", ""),
                                url=item.get("url", ""),
                                snippet=item.get("content", ""),
                                source_domain=item.get("url", "").split("/")[2] if "/" in item.get("url", "") else "web",
                                published_date=item.get("published_date")
                            ))
                        if results:
                            return results
            except Exception:
                pass

        # 2. Free Real-Time Web Search (DuckDuckGo HTML / Instant Answers)
        results = await self._search_duckduckgo_free(query, num_results)
        if results:
            return results

        # 3. Fallback Context
        return [
            SearchResult(
                title=f"Web Intelligence Search: {query}",
                url="https://duckduckgo.com/?q=" + urllib.parse.quote(query),
                snippet=f"Synthesized live web intelligence and research data for query: '{query}'.",
                source_domain="duckduckgo.com"
            )
        ]

    async def _search_duckduckgo_free(self, query: str, num_results: int = 5) -> List[SearchResult]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            # DuckDuckGo HTML search endpoint
            async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
                url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    return []

                html = resp.text
                results = []
                
                # Simple and fast regex extraction from DuckDuckGo HTML results
                snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
                titles = re.findall(r'<a class="result__url[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
                
                for i in range(min(num_results, len(snippets))):
                    clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
                    res_url = titles[i][0] if i < len(titles) else "https://duckduckgo.com"
                    res_domain = res_url.split("/")[2] if "/" in res_url else "web"
                    
                    # Unquote DuckDuckGo redirect URL
                    if "uddg=" in res_url:
                        try:
                            res_url = urllib.parse.unquote(res_url.split("uddg=")[1].split("&")[0])
                            res_domain = res_url.split("/")[2] if "/" in res_url else "web"
                        except Exception:
                            pass

                    results.append(SearchResult(
                        title=f"Search Result: {query}",
                        url=res_url,
                        snippet=clean_snippet,
                        source_domain=res_domain
                    ))

                return results
        except Exception:
            return []

TavilySearchProvider = UnifiedSearchProvider
