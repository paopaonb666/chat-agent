"""Tavily 搜索引擎实现。"""

import httpx

from app.services.search_engines import SearchEngine


class TavilyEngine(SearchEngine):
    """通过 Tavily REST API 执行搜索。"""

    name = "tavily"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int) -> list[dict]:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for i, r in enumerate(data.get("results", [])):
            if i >= max_results:
                break
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "position": i + 1,
                "engine": self.name,
            })
        return results
