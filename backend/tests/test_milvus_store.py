import pytest
from unittest.mock import MagicMock, patch
from app.services.milvus_store import get_milvus_client, ensure_collection, insert_message, search_dense


def test_get_milvus_client_uses_env_uri():
    with patch("app.services.milvus_store.MilvusClient") as mock_cls:
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        client = get_milvus_client()
        mock_cls.assert_called_once_with(uri="http://localhost:19530")
        assert client is mock_inst


def test_ensure_collection_creates_when_missing():
    with patch("app.services.milvus_store.MilvusClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_cls.return_value = mock_client
        ensure_collection(mock_client, dim=768)
        assert mock_client.create_collection.called
        assert mock_client.create_index.called


def test_insert_message_calls_upsert():
    mock_client = MagicMock()
    insert_message(
        mock_client,
        conversation_id="abc",
        user_id=1,
        role="user",
        content="hello",
        message_id=10,
        dense_embedding=[0.1] * 768,
    )
    assert mock_client.insert.called


def test_search_dense_calls_search():
    mock_client = MagicMock()
    mock_client.search.return_value = [
        [{"id": 1, "distance": 0.9, "entity": {"conversation_id": "c1", "role": "user", "content": "hi", "message_id": 1, "timestamp": 0}}]
    ]
    results = search_dense(mock_client, dense_embedding=[0.1] * 768, user_id=1, top_k=5)
    assert mock_client.search.called
    assert len(results) == 1
