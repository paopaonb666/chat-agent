import os
import time
import asyncio
from duckduckgo_search import DDGS

SEARCH_PROXY = os.getenv("SEARCH_PROXY", None)

_cache: dict[str, tuple[float, list[dict]]] = {}
CACHE_TTL = 300


async def web_search(query: str, max_results: int = 8) -> list[dict]:
    now = time.time()
    cached = _cache.get(query)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]

    try:
        results = await asyncio.to_thread(_do_search, query, max_results)
        _cache[query] = (now, results)
        return results
    except Exception:
        return []


def _do_search(query: str, max_results: int) -> list[dict]:
    ddgs_kwargs = {}
    if SEARCH_PROXY:
        ddgs_kwargs["proxy"] = SEARCH_PROXY

    with DDGS(**ddgs_kwargs) as ddgs:
        raw = list(ddgs.text(query, max_results=max_results))
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
            "position": i + 1,
        }
        for i, r in enumerate(raw)
    ]


def format_web_context(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["以下是从互联网搜索到的相关信息：\n"]
    for s in sources:
        lines.append(f"[{s['position']}] {s['title']}")
        lines.append(f"    URL: {s['url']}")
        lines.append(f"    摘要: {s['snippet']}")
        lines.append("")
    return "\n".join(lines)


def clear_web_cache() -> None:
    _cache.clear()
