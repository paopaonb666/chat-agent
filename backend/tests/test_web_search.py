import pytest
from unittest.mock import patch, AsyncMock

from app.services.web_search import web_search, format_web_context


@pytest.mark.asyncio
async def test_web_search_returns_structured_results():
    mock_results = [
        {"title": "Python 教程", "url": "https://example.com/python", "snippet": "Python 入门指南", "position": 1, "engine": "tavily"},
        {"title": "Python 进阶", "url": "https://example.com/python2", "snippet": "高级特性", "position": 2, "engine": "tavily"},
    ]
    mock_manager = AsyncMock()
    mock_manager.async_search.return_value = mock_results
    with patch("app.services.web_search._get_search_manager", return_value=mock_manager):
        results = await web_search("Python 教程")
        assert len(results) == 2
        assert results[0]["title"] == "Python 教程"
        assert results[0]["url"] == "https://example.com/python"
        assert results[0]["snippet"] == "Python 入门指南"
        assert results[0]["position"] == 1
        assert results[0]["engine"] == "tavily"


@pytest.mark.asyncio
async def test_web_search_empty_on_exception():
    mock_manager = AsyncMock()
    mock_manager.async_search.side_effect = Exception("API error")
    with patch("app.services.web_search._get_search_manager", return_value=mock_manager):
        results = await web_search("test")
        assert results == []


@pytest.mark.asyncio
async def test_web_search_cache():
    mock_results = [{"title": "Cached", "url": "https://c.com", "snippet": "Cache test", "position": 1, "engine": "duckduckgo"}]
    mock_manager = AsyncMock()
    mock_manager.async_search.return_value = mock_results
    with patch("app.services.web_search._get_search_manager", return_value=mock_manager):
        # First call
        results1 = await web_search("cache test")
        # Second call should use cache (no second async_search call)
        results2 = await web_search("cache test")
        assert results1 == results2
        from app.services.web_search import _cache
        assert "cache test" in _cache
        # Manager only called once
        assert mock_manager.async_search.call_count == 1


def test_format_web_context():
    sources = [
        {"title": "T1", "url": "https://a.com", "snippet": "S1", "position": 1, "engine": "tavily"},
        {"title": "T2", "url": "https://b.com", "snippet": "S2", "position": 2, "engine": "duckduckgo"},
    ]
    text = format_web_context(sources)
    assert "[1]" in text
    assert "T1" in text
    assert "https://a.com" in text
    assert "[2]" in text
    assert "来源: tavily" in text
    assert "来源: duckduckgo" in text


def test_format_web_context_empty():
    assert format_web_context([]) == ""


@pytest.mark.asyncio
async def test_web_search_enhances_bad_query():
    mock_results = [
        {"title": "拉康 精神分析", "url": "https://example.com/lacan", "snippet": "拉康理论", "position": 1, "engine": "tavily"},
    ]
    mock_manager = AsyncMock()
    mock_manager.async_search.return_value = mock_results
    with patch("app.services.web_search._get_search_manager", return_value=mock_manager):
        results = await web_search("拉", user_message="拉康的理论是什么")
        called_query = mock_manager.async_search.call_args[0][0]
        assert "拉康" in called_query


@pytest.mark.asyncio
async def test_web_search_filters_low_quality():
    from app.services.web_search import _score_and_filter_results

    raw = [
        {"title": "拉的字义", "url": "https://www.zdic.net/hans/拉", "snippet": "汉字解释", "position": 1, "engine": "duckduckgo"},
        {"title": "拉康 - 维基百科", "url": "https://zh.wikipedia.org/wiki/拉康", "snippet": "雅克·拉康", "position": 2, "engine": "tavily"},
    ]
    filtered = _score_and_filter_results(raw, "拉康 精神分析")
    assert len(filtered) == 1
    assert "维基百科" in filtered[0]["title"]


@pytest.mark.asyncio
async def test_web_search_degrades_on_failure():
    """搜索失败时返回空列表（降级逻辑在 FallbackSearchManager 中处理）。"""
    mock_manager = AsyncMock()
    mock_manager.async_search.side_effect = Exception("timeout")
    with patch("app.services.web_search._get_search_manager", return_value=mock_manager):
        results = await web_search("test", user_message="test message")
        assert results == []
        assert mock_manager.async_search.call_count == 1
