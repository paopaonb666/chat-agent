from unittest.mock import patch, MagicMock
import app.core.milvus as milvus_core


def test_get_milvus_client_returns_singleton():
    milvus_core._milvus_client = None
    with patch("app.core.milvus.MilvusClient") as mock_cls:
        instance1 = MagicMock()
        mock_cls.return_value = instance1
        client1 = milvus_core.get_milvus_client()
        client2 = milvus_core.get_milvus_client()
        assert client1 is client2
        mock_cls.assert_called_once()


def test_get_milvus_client_uses_configured_uri():
    milvus_core._milvus_client = None
    with patch("app.core.milvus.MilvusClient") as mock_cls:
        milvus_core.get_milvus_client()
        _, kwargs = mock_cls.call_args
        assert "uri" in kwargs
