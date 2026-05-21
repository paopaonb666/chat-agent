import pytest
from unittest.mock import MagicMock, patch
import app.core.milvus as milvus_core
from app.core.milvus import get_milvus_client
from app.services.milvus_store import ensure_collection, insert_message, search_dense


def test_get_milvus_client_uses_env_uri():
    milvus_core._milvus_client = None
    with patch("app.core.milvus.MilvusClient") as mock_cls:
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        client = get_milvus_client()
        mock_cls.assert_called_once_with(uri="http://localhost:19530")
        assert client is mock_inst


def test_ensure_collection_creates_when_missing():
    with patch("app.core.milvus.MilvusClient") as mock_cls:
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


def test_ensure_collection_drops_and_recreate_on_dim_mismatch():
    """当已有 collection 的维度与目标维度不一致时，应删除重建。"""
    mock_client = MagicMock()
    mock_client.list_collections.return_value = ["conversation_history"]
    mock_client.describe_collection.return_value = {
        "fields": [
            {"name": "id", "params": {}},
            {"name": "dense_embedding", "params": {"dim": 768}},  # 旧维度
        ]
    }
    ensure_collection(mock_client, dim=1024)
    mock_client.drop_collection.assert_called_once_with("conversation_history")
    mock_client.create_collection.assert_called_once()
    mock_client.create_index.assert_called_once()
