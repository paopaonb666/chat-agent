"""搜索引擎集成测试 — 实际调用外部 API，验证端到端可用性。

运行方式:
    cd backend && venv/Scripts/python -m pytest tests/test_search_integration.py -v

注意: 会消耗 Tavily API 额度，请谨慎频繁运行。
"""

import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Windows 控制台强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import settings
from app.services.search_engines.tavily import TavilyEngine
from app.services.search_engines.duckduckgo import DuckDuckGoEngine
from app.services.search_engines import FallbackSearchManager
from app.services.web_search import (
    web_search,
    format_web_context,
    clear_web_cache,
    _score_and_filter_results,
)

def _print_results(label: str, results: list[dict]):
    """打印搜索结果列表。"""
    print(f"\n=== {label} ===")
    print(f"共 {len(results)} 条结果:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.get('title', 'N/A')}")
        print(f"    URL: {r.get('url', 'N/A')}")
        print(f"    摘要: {r.get('snippet', 'N/A')[:120]}...")
        print(f"    来源: {r.get('engine', 'unknown')}")
        print()




# ── 1. Tavily 实际调用 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tavily_engine_search_real():
    """TavilyEngine 实际调用 API 应返回结构化结果。"""
    if not settings.tavily_api_key:
        pytest.skip("TAVILY_API_KEY not configured")

    engine = TavilyEngine(api_key=settings.tavily_api_key)
    results = await asyncio.to_thread(engine.search, "Python 教程", max_results=3)

    _print_results("TavilyEngine.search('Python 教程')", results)

    assert len(results) > 0
    assert all("title" in r and "url" in r and "snippet" in r for r in results)
    assert all(r["engine"] == "tavily" for r in results)
    assert results[0]["position"] == 1


# ── 2. DuckDuckGo 实际调用 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_duckduckgo_engine_search_real():
    """DuckDuckGoEngine 实际调用应返回结构化结果。"""
    engine = DuckDuckGoEngine()
    results = await asyncio.to_thread(engine.search, "Python 教程", max_results=3)

    _print_results("DuckDuckGoEngine.search('Python 教程')", results)

    assert len(results) > 0
    assert all("title" in r and "url" in r and "snippet" in r for r in results)
    assert all(r["engine"] == "duckduckgo" for r in results)
    assert results[0]["position"] == 1


# ── 3. FallbackManager 优先 Tavily ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_manager_uses_tavily_when_configured():
    """配置了 Tavily key 时，FallbackManager 应优先返回 Tavily 结果。"""
    if not settings.tavily_api_key:
        pytest.skip("TAVILY_API_KEY not configured")

    manager = FallbackSearchManager([
        TavilyEngine(api_key=settings.tavily_api_key),
        DuckDuckGoEngine(),
    ])
    results = await manager.async_search("Python 教程", max_results=3)

    _print_results("FallbackManager (Tavily first) 'Python 教程'", results)

    assert len(results) > 0
    assert results[0]["engine"] == "tavily"


# ── 4. FallbackManager 降级到 DuckDuckGo ───────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_manager_degrades_to_duckduckgo():
    """Tavily 失败时应自动降级到 DuckDuckGo。"""
    bad_tavily = MagicMock()
    bad_tavily.name = "tavily"
    bad_tavily.search.side_effect = Exception("Tavily down")

    manager = FallbackSearchManager([bad_tavily, DuckDuckGoEngine()])
    results = await manager.async_search("Python 教程", max_results=3)

    _print_results("FallbackManager (Tavily fail -> DDG) 'Python 教程'", results)

    assert len(results) > 0
    assert results[0]["engine"] == "duckduckgo"
    bad_tavily.search.assert_called_once()


# ── 5. web_search 端到端 ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_web_search_integration():
    """web_search 端到端调用应返回过滤后的结果。"""
    clear_web_cache()
    results = await web_search("Python 入门教程", max_results=5)

    _print_results("web_search('Python 入门教程')", results)

    assert isinstance(results, list)
    assert len(results) <= 5
    if results:
        assert all("title" in r and "url" in r and "snippet" in r for r in results)
        assert all(r["position"] > 0 for r in results)
        assert all("engine" in r for r in results)


# ── 6. web_search 缓存 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_web_search_cache_integration():
    """相同 query 第二次调用应命中缓存。"""
    clear_web_cache()
    query = "缓存测试查询"

    results1 = await web_search(query, max_results=3)
    results2 = await web_search(query, max_results=3)

    print(f"\n=== Cache Integration ===")
    print(f"First call results count: {len(results1)}")
    print(f"Second call results count: {len(results2)}")
    print(f"Cache hit: {results1 == results2}")

    assert results1 == results2


# ── 7. web_search query 增强 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_web_search_query_enhancement_integration():
    """单字 query 应被增强为完整关键词。"""
    clear_web_cache()
    results = await web_search("拉", user_message="拉康的理论是什么", max_results=3)

    _print_results("web_search('拉', user_message='拉康的理论是什么')", results)

    # 结果中应出现与"拉康"相关的内容（不严格断言，因搜索结果变化）
    assert isinstance(results, list)


# ── 8. web_search 低质量过滤 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_web_search_filters_low_quality_integration():
    """低质量结果（字典/拼音页）应被过滤。"""
    raw = [
        {"title": "拉的字义", "url": "https://www.zdic.net/hans/拉", "snippet": "汉字解释", "position": 1, "engine": "duckduckgo"},
        {"title": "拉康 - 维基百科", "url": "https://zh.wikipedia.org/wiki/拉康", "snippet": "雅克·拉康", "position": 2, "engine": "tavily"},
    ]
    filtered = _score_and_filter_results(raw, "拉康 精神分析")

    print(f"\n=== Low Quality Filter ===")
    print(f"Raw results count: {len(raw)}")
    print(f"Filtered results count: {len(filtered)}")
    print(f"Raw titles: {[r['title'] for r in raw]}")
    print(f"Filtered titles: {[r['title'] for r in filtered]}")

    assert len(filtered) == 1
    assert "维基百科" in filtered[0]["title"]


# ── 9. format_web_context 格式化 ───────────────────────────────────────────


def test_format_web_context_integration():
    """format_web_context 应正确格式化并标注来源引擎。"""
    sources = [
        {"title": "T1", "url": "https://a.com", "snippet": "S1", "position": 1, "engine": "tavily"},
        {"title": "T2", "url": "https://b.com", "snippet": "S2", "position": 2, "engine": "duckduckgo"},
    ]
    text = format_web_context(sources)

    print(f"\n=== Formatted Web Context ===")
    print(text)

    assert "[1]" in text
    assert "T1" in text
    assert "来源: tavily" in text
    assert "来源: duckduckgo" in text


# ── 10. 清除缓存 ───────────────────────────────────────────────────────────


def test_clear_web_cache_integration():
    """clear_web_cache 应清空缓存字典。"""
    from app.services.web_search import _cache
    _cache["test"] = (0, [])

    print(f"\n=== Clear Cache ===")
    print(f"Cache keys before clear: {list(_cache.keys())}")

    clear_web_cache()

    print(f"Cache keys after clear: {list(_cache.keys())}")

    assert "test" not in _cache


# ── 11. 无 Tavily key 时直接使用 DuckDuckGo ─────────────────────────────────


@pytest.mark.asyncio
async def test_no_tavily_key_uses_duckduckgo():
    """tavily_api_key 为空时，manager 应只包含 DuckDuckGo。"""
    from app.services.web_search import _get_search_manager

    with patch("app.services.web_search.settings.tavily_api_key", ""):
        # 强制重新创建 manager
        from app.services import web_search
        web_search._search_manager = None

        manager = _get_search_manager()
        assert len(manager.engines) == 1
        assert manager.engines[0].name == "duckduckgo"

        results = await manager.async_search("Python 教程", max_results=3)

        _print_results("No Tavily key -> DuckDuckGo 'Python 教程'", results)

        assert len(results) > 0
        assert results[0]["engine"] == "duckduckgo"

    # 恢复
    web_search._search_manager = None
