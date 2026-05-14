import pytest
from unittest.mock import patch, AsyncMock
from app.services.rag_pipeline import run_rag


@pytest.mark.asyncio
async def test_run_rag_returns_context_when_retrieval_needed():
    with patch("app.services.rag_pipeline.recognize_intent", new_callable=AsyncMock) as mock_intent, \
         patch("app.services.rag_pipeline.get_dense_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.services.rag_pipeline.hybrid_search", new_callable=AsyncMock) as mock_hybrid, \
         patch("app.services.rag_pipeline.rerank_passages", new_callable=AsyncMock) as mock_rerank:
        mock_intent.return_value.needs_retrieval = True
        mock_intent.return_value.refined_query = "Python 错误"
        mock_emb.return_value = [0.1] * 768
        mock_hybrid.return_value = [{"content": "try except"}]
        mock_rerank.return_value = [{"content": "try except"}]
        context = await run_rag("Python报错怎么办", "conv-1", user_id=1, messages=[])
        assert "try except" in context
        assert "历史对话" in context


@pytest.mark.asyncio
async def test_run_rag_empty_when_no_retrieval():
    with patch("app.services.rag_pipeline.recognize_intent", new_callable=AsyncMock) as mock_intent:
        mock_intent.return_value.needs_retrieval = False
        context = await run_rag("你好", "conv-1", user_id=1, messages=[])
        assert context == ""
