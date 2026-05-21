"""DuckDuckGo 搜索引擎实现。"""

from ddgs import DDGS

from app.services.search_engines import SearchEngine


class DuckDuckGoEngine(SearchEngine):
    """通过 DuckDuckGo 免费搜索 API 执行搜索。"""

    name = "duckduckgo"

    def search(self, query: str, max_results: int) -> list[dict]:
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=max_results)

        results = []
        for i, r in enumerate(raw):
            if i >= max_results:
                break
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "position": i + 1,
                "engine": self.name,
            })
        return results
