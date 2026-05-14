import pytest
from unittest.mock import patch, AsyncMock
from app.services.intent import recognize_intent, IntentResult


@pytest.mark.asyncio
async def test_recognize_intent_needs_retrieval():
    mock_resp = {
        "message": {"content": '{"needs_retrieval": true, "refined_query": "如何解决Python报错", "reason": "用户询问技术问题"}'}
    }
    with patch("app.services.intent.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json = lambda: mock_resp
        mock_post.return_value.raise_for_status = lambda: None
        intent = await recognize_intent("Python报错了怎么办", [])
        assert intent.needs_retrieval is True
        assert intent.refined_query == "如何解决Python报错"


@pytest.mark.asyncio
async def test_recognize_intent_no_retrieval():
    mock_resp = {
        "message": {"content": '{"needs_retrieval": false, "refined_query": "", "reason": "闲聊"}'}
    }
    with patch("app.services.intent.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json = lambda: mock_resp
        mock_post.return_value.raise_for_status = lambda: None
        intent = await recognize_intent("你好", [])
        assert intent.needs_retrieval is False
