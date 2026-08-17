import re
import asyncio
from typing import Dict, Any, List, Optional
import httpx

class BrowserAgent:
    """
    Autonomous Web Browser Agent.
    Navigates pages, extracts clean content, parses structured data (tables, lists)
    without requiring heavy browser automation dependencies (pure HTTP + HTML parsing).
    """
    def __init__(self):
        self.session_history: List[str] = []
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8"
        }

    async def fetch_url(self, url: str, timeout: float = 8.0) -> Dict[str, Any]:
        """Fetch and parse a web page, returning clean extracted text and metadata."""
        if not url.startswith("http"):
            url = "https://" + url
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=self.headers)
                html = response.text
                final_url = str(response.url)
                status_code = response.status_code
        except httpx.TimeoutException:
            return {"status": "error", "error": "Request timed out", "url": url}
        except Exception as e:
            return {"status": "error", "error": str(e), "url": url}

        clean_text = self._clean_html(html)
        title = self._extract_title(html)
        links = self._extract_links(html, base_url=final_url)
        tables = self._extract_tables(html)

        self.session_history.append(final_url)

        return {
            "status": "success",
            "url": final_url,
            "title": title,
            "text": clean_text[:5000],
            "links": links[:10],
            "tables": tables[:3],
            "http_status": status_code
        }

    def _clean_html(self, html: str) -> str:
        """Strip HTML tags, scripts, styles and collapse whitespace."""
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<[^>]+>', ' ', html)
        html = re.sub(r'&nbsp;', ' ', html)
        html = re.sub(r'&[a-z]+;', '', html)
        html = re.sub(r'\s+', ' ', html).strip()
        return html

    def _extract_title(self, html: str) -> str:
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        return re.sub(r'\s+', ' ', match.group(1)).strip() if match else "No title"

    def _extract_links(self, html: str, base_url: str) -> List[Dict[str, str]]:
        links = []
        for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
            href = m.group(1).strip()
            text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
            if href.startswith("http") and text:
                links.append({"href": href, "text": text[:60]})
        return links

    def _extract_tables(self, html: str) -> List[List[List[str]]]:
        """Parse HTML tables into 2D lists."""
        tables = []
        for table_html in re.findall(r'<table[^>]*>(.*?)</table>', html, re.DOTALL | re.IGNORECASE):
            rows = []
            for row_html in re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE):
                cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)
                row = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                if any(row):
                    rows.append(row)
            if rows:
                tables.append(rows)
        return tables

    async def execute_action(self, action: str, url: str = "", data: Dict = None) -> Dict[str, Any]:
        """
        Execute a structured browser action.
        Supported actions: 'navigate', 'extract_text', 'find_links', 'extract_tables'
        """
        if action in ("navigate", "extract_text", "find_links", "extract_tables"):
            result = await self.fetch_url(url)
            if action == "find_links":
                return {"links": result.get("links", []), "status": "success"}
            if action == "extract_tables":
                return {"tables": result.get("tables", []), "status": "success"}
            return result
        return {"status": "error", "error": f"Unknown action: {action}"}

browser_agent = BrowserAgent()
