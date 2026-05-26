import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.knowledge_rag import search_knowledge_base_rag


@pytest.mark.asyncio
async def test_kb_rag_returns_formatted_context():
    mock_doc = MagicMock()
    mock_doc.id = "doc-1"
    mock_doc.visibility = "public"

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_doc]

    with patch("app.services.knowledge_rag.get_dense_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.services.knowledge_rag.get_milvus_client") as mock_milvus, \
         patch("app.services.knowledge_rag.search_knowledge_base") as mock_search:
        mock_emb.return_value = [0.1] * 1024
        mock_search.return_value = [
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "content": "Python supports async/await.",
                "distance": 0.85,
                "title_path": "Chapter 1 > Intro",
            }
        ]
        result = await search_knowledge_base_rag(mock_db, "Python async", user_id=1, top_k=5)
        assert "Python supports async/await" in result
        assert "Chapter 1 > Intro" in result


@pytest.mark.asyncio
async def test_kb_rag_returns_empty_when_no_visible_docs():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = []

    result = await search_knowledge_base_rag(mock_db, "query", user_id=1, top_k=5)
    assert result == ""


@pytest.mark.asyncio
async def test_kb_rag_returns_empty_on_search_failure():
    mock_doc = MagicMock()
    mock_doc.id = "doc-1"
    mock_doc.visibility = "public"

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_doc]

    with patch("app.services.knowledge_rag.get_dense_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.services.knowledge_rag.get_milvus_client") as mock_milvus, \
         patch("app.services.knowledge_rag.search_knowledge_base") as mock_search:
        mock_emb.return_value = [0.1] * 1024
        mock_search.side_effect = Exception("Milvus down")
        result = await search_knowledge_base_rag(mock_db, "query", user_id=1, top_k=5)
        assert result == ""
