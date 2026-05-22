import time
import asyncio
import logging
from app.core.config import settings
from app.services.query_validator import (
    enhance_query,
    is_low_quality_result,
)
from app.services.search_engines import FallbackSearchManager
from app.services.search_engines.tavily import TavilyEngine
from app.services.search_engines.duckduckgo import DuckDuckGoEngine

logger = logging.getLogger(__name__)

SEARCH_TIMEOUT = 10.0

_cache: dict[str, tuple[float, list[dict]]] = {}
CACHE_TTL = 300

_search_manager: FallbackSearchManager | None = None


def _get_search_manager() -> FallbackSearchManager:
    """懒加载搜索管理器单例，按配置顺序组装引擎列表。"""
    global _search_manager
    if _search_manager is not None:
        return _search_manager

    engines = []
    priority = [e.strip().lower() for e in settings.search_engine_priority.split(",")]

    for name in priority:
        if name == "tavily" and settings.tavily_api_key:
            engines.append(TavilyEngine(api_key=settings.tavily_api_key))
        elif name == "duckduckgo":
            engines.append(DuckDuckGoEngine())

    if not engines:
        logger.warning("No search engines configured, falling back to DuckDuckGo")
        engines.append(DuckDuckGoEngine())

    _search_manager = FallbackSearchManager(engines)
    return _search_manager


async def web_search(query: str, max_results: int = 8, user_message: str = "") -> list[dict]:
    """执行网络搜索，带 query 校验、结果过滤和自动降级。

    Args:
        query: 原始搜索 query（可能来自 LLM）
        max_results: 最大返回结果数
        user_message: 用户原始消息，用于 query 增强的上下文
    """
    # 1. Query 校验与增强
    validation = await enhance_query(query, user_message)
    search_query = validation.query
    if validation.was_enhanced:
        logger.info("Query enhanced: '%s' -> '%s' (%s)", query, search_query, validation.reason)

    # 2. 缓存检查（用增强后的 query）
    now = time.time()
    cached = _cache.get(search_query)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]

    # 3. 搜索（自动降级）
    manager = _get_search_manager()
    try:
        results = await asyncio.wait_for(
            manager.async_search(search_query, max_results * 2),
            timeout=SEARCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Web search timed out after %.1fs", SEARCH_TIMEOUT)
        results = []
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        results = []

    if not results:
        return []

    # 4. 质量过滤与排序
    scored = _score_and_filter_results(results, search_query)
    final = scored[:max_results]
    # 重新编号
    for i, r in enumerate(final, start=1):
        r["position"] = i

    # 5. 缓存
    if final:
        _cache[search_query] = (now, final)

    return final


def _score_and_filter_results(results: list[dict], query: str) -> list[dict]:
    """对搜索结果评分、过滤低质量结果并按相关性排序。"""
    query_terms = [t for t in query.lower().split() if len(t) > 1]
    scored = []
    for r in results:
        if is_low_quality_result(r):
            logger.debug("Filtered low-quality result: %s", r.get("url"))
            continue
        score = _score_relevance(r, query_terms)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


def _score_relevance(result: dict, query_terms: list[str]) -> float:
    """基于 query 关键词在 title 和 snippet 中的匹配度评分。"""
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    text = title + " " + snippet
    if not query_terms:
        return 0.0

    matches = sum(1 for t in query_terms if t in text)
    # 标题匹配权重更高
    title_matches = sum(1 for t in query_terms if t in title)
    return matches + title_matches * 2.0


def format_web_context(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["以下是从互联网搜索到的相关信息：\n"]
    for s in sources:
        lines.append(f"[{s['position']}] {s['title']}")
        lines.append(f"    URL: {s['url']}")
        lines.append(f"    摘要: {s['snippet']} (来源: {s.get('engine', 'unknown')})")
        lines.append("")
    return "\n".join(lines)


def clear_web_cache() -> None:
    _cache.clear()
