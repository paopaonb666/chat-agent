import pytest
from unittest.mock import patch
from app.services.web_search import web_search, format_web_context


@pytest.mark.asyncio
async def test_web_search_returns_structured_results():
    mock_results = [
        {"title": "Python 教程", "href": "https://example.com/python", "body": "Python 入门指南"},
        {"title": "Python 进阶", "href": "https://example.com/python2", "body": "高级特性"},
    ]
    with patch("app.services.web_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = mock_results
        results = await web_search("Python 教程")
        assert len(results) == 2
        assert results[0]["title"] == "Python 教程"
        assert results[0]["url"] == "https://example.com/python"
        assert results[0]["snippet"] == "Python 入门指南"
        assert results[0]["position"] == 1


@pytest.mark.asyncio
async def test_web_search_empty_on_exception():
    with patch("app.services.web_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.side_effect = Exception("API error")
        results = await web_search("test")
        assert results == []


def test_format_web_context():
    sources = [
        {"title": "T1", "url": "https://a.com", "snippet": "S1", "position": 1},
        {"title": "T2", "url": "https://b.com", "snippet": "S2", "position": 2},
    ]
    text = format_web_context(sources)
    assert "[1]" in text
    assert "T1" in text
    assert "https://a.com" in text
    assert "[2]" in text
