import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.rerank import rerank_passages


def _mock_chat_response(score_text: str):
    """Build a fake httpx response in /api/chat format."""
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": score_text}}
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_rerank_passages_sorts_by_score():
    passages = [
        {"id": 1, "content": "Python 教程"},
        {"id": 2, "content": "Java 教程"},
    ]
    mock_responses = [_mock_chat_response("9.5"), _mock_chat_response("3.0")]
    with patch("app.services.rerank.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = mock_responses
        results = await rerank_passages("Python 学习", passages, top_n=2)
        assert results[0]["id"] == 1
        assert len(results) == 2


@pytest.mark.asyncio
async def test_rerank_handles_http_error_gracefully():
    passages = [
        {"id": 1, "content": "Python 教程"},
        {"id": 2, "content": "Java 教程"},
    ]
    good = _mock_chat_response("7.0")
    bad = MagicMock()
    bad.raise_for_status.side_effect = Exception("HTTP 500")
    with patch("app.services.rerank.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [good, bad]
        results = await rerank_passages("Python 学习", passages, top_n=2)
        assert results[0]["id"] == 1
        assert len(results) == 2


@pytest.mark.asyncio
async def test_rerank_empty_passages():
    results = await rerank_passages("test", [], top_n=5)
    assert results == []


@pytest.mark.asyncio
async def test_rerank_respects_top_n():
    passages = [{"id": i, "content": f"doc{i}"} for i in range(10)]
    mocks = [_mock_chat_response(str(10 - i)) for i in range(10)]
    with patch("app.services.rerank.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = mocks
        results = await rerank_passages("query", passages, top_n=3)
        assert len(results) == 3
        # Highest score first
        assert results[0]["id"] == 0
