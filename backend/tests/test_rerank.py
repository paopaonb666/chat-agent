import pytest
from unittest.mock import patch, AsyncMock
from app.services.rerank import rerank_passages


@pytest.mark.asyncio
async def test_rerank_passages_sorts_by_score():
    passages = [
        {"id": 1, "content": "Python 教程"},
        {"id": 2, "content": "Java 教程"},
    ]
    responses = [
        {"response": "9.5"},
        {"response": "3.0"},
    ]
    with patch("app.services.rerank.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [
            AsyncMock(json=lambda: r, raise_for_status=lambda: None) for r in responses
        ]
        results = await rerank_passages("Python 学习", passages, top_n=2)
        assert results[0]["id"] == 1
        assert len(results) == 2
