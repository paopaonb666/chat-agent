"""搜索引擎模块测试 — TDD 先写测试再实现。"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.search_engines import FallbackSearchManager
from app.services.search_engines.tavily import TavilyEngine
from app.services.search_engines.duckduckgo import DuckDuckGoEngine


# ── TavilyEngine 测试 ─────────────────────────────────────────────────────


def test_tavily_engine_search():
    """TavilyEngine 应正确解析 API 响应并返回统一格式。"""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"title": "T1", "url": "https://a.com", "content": "Snippet 1"},
            {"title": "T2", "url": "https://b.com", "content": "Snippet 2"},
        ]
    }

    with patch("app.services.search_engines.tavily.httpx.post", return_value=mock_resp):
        engine = TavilyEngine(api_key="test-key")
        results = engine.search("python", max_results=5)

    assert len(results) == 2
    assert results[0]["title"] == "T1"
    assert results[0]["url"] == "https://a.com"
    assert results[0]["snippet"] == "Snippet 1"
    assert results[0]["position"] == 1
    assert results[0]["engine"] == "tavily"


def test_tavily_engine_raises_on_error():
    """Tavily API 失败时应抛出异常以便 manager 降级。"""
    with patch("app.services.search_engines.tavily.httpx.post", side_effect=Exception("API error")):
        engine = TavilyEngine(api_key="test-key")
        with pytest.raises(Exception):
            engine.search("python", max_results=5)


# ── DuckDuckGoEngine 测试 ─────────────────────────────────────────────────


def test_duckduckgo_engine_search():
    """DuckDuckGoEngine 应正确解析 DDGS 结果并返回统一格式。"""
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {"title": "D1", "href": "https://d.com", "body": "Body 1"},
        {"title": "D2", "href": "https://e.com", "body": "Body 2"},
    ]
    mock_ddgs.__enter__.return_value = mock_ddgs

    with patch("app.services.search_engines.duckduckgo.DDGS", return_value=mock_ddgs):
        engine = DuckDuckGoEngine()
        results = engine.search("python", max_results=5)

    assert len(results) == 2
    assert results[0]["title"] == "D1"
    assert results[0]["url"] == "https://d.com"
    assert results[0]["snippet"] == "Body 1"
    assert results[0]["position"] == 1
    assert results[0]["engine"] == "duckduckgo"


# ── FallbackSearchManager 测试 ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_manager_uses_first_successful():
    """第一个引擎成功时应立即返回其结果，不尝试后续引擎。"""
    engine1 = MagicMock()
    engine1.name = "engine1"
    engine1.search.return_value = [
        {"title": "E1", "url": "https://e1.com", "snippet": "S1", "position": 1, "engine": "engine1"}
    ]
    engine2 = MagicMock()
    engine2.name = "engine2"

    manager = FallbackSearchManager([engine1, engine2])
    results = await manager.async_search("query", max_results=3)

    assert len(results) == 1
    assert results[0]["engine"] == "engine1"
    engine1.search.assert_called_once_with("query", 3)
    engine2.search.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_manager_falls_back_to_second():
    """第一个引擎失败时应自动降级到第二个引擎。"""
    engine1 = MagicMock()
    engine1.name = "engine1"
    engine1.search.side_effect = Exception("fail")
    engine2 = MagicMock()
    engine2.name = "engine2"
    engine2.search.return_value = [
        {"title": "E2", "url": "https://e2.com", "snippet": "S2", "position": 1, "engine": "engine2"}
    ]

    manager = FallbackSearchManager([engine1, engine2])
    results = await manager.async_search("query", max_results=3)

    assert len(results) == 1
    assert results[0]["engine"] == "engine2"
    engine1.search.assert_called_once()
    engine2.search.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_manager_returns_empty_when_all_fail():
    """所有引擎都失败时应返回空列表。"""
    engine1 = MagicMock()
    engine1.name = "engine1"
    engine1.search.side_effect = Exception("fail1")
    engine2 = MagicMock()
    engine2.name = "engine2"
    engine2.search.side_effect = Exception("fail2")

    manager = FallbackSearchManager([engine1, engine2])
    results = await manager.async_search("query", max_results=3)

    assert results == []
