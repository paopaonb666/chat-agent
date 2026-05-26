import pytest
from unittest.mock import patch, AsyncMock
from app.services.rag_pipeline import run_rag


@pytest.mark.asyncio
async def test_run_rag_returns_context_when_retrieval_needed():
    with patch("app.services.rag_pipeline.get_dense_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.services.rag_pipeline.hybrid_search", new_callable=AsyncMock) as mock_hybrid, \
         patch("app.services.rag_pipeline.rerank_passages", new_callable=AsyncMock) as mock_rerank:
        mock_emb.return_value = [0.1] * 768
        mock_hybrid.return_value = [{"content": "try except"}]
        mock_rerank.return_value = [{"content": "try except"}]
        mock_db = object()
        context = await run_rag(mock_db, "Python报错怎么办", "conv-1", user_id=1, messages=[],
                                query_override="Python 错误")
        assert "try except" in context
        assert "历史对话" in context


@pytest.mark.asyncio
async def test_run_rag_empty_when_no_retrieval():
    with patch("app.services.rag_pipeline.get_dense_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.services.rag_pipeline.hybrid_search", new_callable=AsyncMock) as mock_hybrid:
        mock_hybrid.return_value = []
        context = await run_rag(None, "你好", "conv-1", user_id=1, messages=[],
                                query_override="你好")
        assert context == ""


@pytest.mark.asyncio
async def test_run_rag_skips_rerank_when_few_results():
    """When hybrid_search returns < 3 results, rerank_passages is NOT called."""
    with patch("app.services.rag_pipeline.get_dense_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.services.rag_pipeline.hybrid_search", new_callable=AsyncMock) as mock_hybrid, \
         patch("app.services.rag_pipeline.rerank_passages", new_callable=AsyncMock) as mock_rerank:
        mock_emb.return_value = [0.1] * 768
        mock_hybrid.return_value = [{"content": "only one result"}]
        context = await run_rag(None, "test", "conv-1", user_id=1, messages=[],
                                query_override="test")
        assert "only one result" in context
        mock_rerank.assert_not_called()


@pytest.mark.asyncio
async def test_run_rag_calls_rerank_when_sufficient_results():
    """When hybrid_search returns >= 3 results, rerank_passages IS called."""
    with patch("app.services.rag_pipeline.get_dense_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.services.rag_pipeline.hybrid_search", new_callable=AsyncMock) as mock_hybrid, \
         patch("app.services.rag_pipeline.rerank_passages", new_callable=AsyncMock) as mock_rerank:
        mock_emb.return_value = [0.1] * 768
        mock_hybrid.return_value = [
            {"content": "a"}, {"content": "b"}, {"content": "c"},
        ]
        mock_rerank.return_value = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        context = await run_rag(None, "test", "conv-1", user_id=1, messages=[],
                                query_override="test")
        assert "a" in context
        mock_rerank.assert_called_once()


@pytest.mark.asyncio
async def test_run_rag_kb_hit_short_circuits():
    """When knowledge base has results, return them directly without hybrid_search."""
    with patch("app.services.rag_pipeline.search_knowledge_base_rag", new_callable=AsyncMock) as mock_kb, \
         patch("app.services.rag_pipeline.get_dense_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.services.rag_pipeline.hybrid_search", new_callable=AsyncMock) as mock_hybrid:
        mock_kb.return_value = "以下是从企业文档知识库中检索到的相关信息：\n[1] KB result"
        context = await run_rag(None, "test", "conv-1", user_id=1, messages=[],
                                query_override="test")
        assert "KB result" in context
        mock_emb.assert_not_called()
        mock_hybrid.assert_not_called()


@pytest.mark.asyncio
async def test_run_rag_kb_miss_falls_back_to_hybrid():
    """When knowledge base returns empty, fall back to hybrid_search."""
    with patch("app.services.rag_pipeline.search_knowledge_base_rag", new_callable=AsyncMock) as mock_kb, \
         patch("app.services.rag_pipeline.get_dense_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.services.rag_pipeline.hybrid_search", new_callable=AsyncMock) as mock_hybrid:
        mock_kb.return_value = ""
        mock_emb.return_value = [0.1] * 768
        mock_hybrid.return_value = [{"content": "fallback result"}]
        context = await run_rag(None, "test", "conv-1", user_id=1, messages=[],
                                query_override="test")
        assert "fallback result" in context
        mock_emb.assert_called_once()
        mock_hybrid.assert_called_once()

