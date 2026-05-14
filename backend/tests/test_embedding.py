import pytest
from unittest.mock import patch, AsyncMock
from app.services.embedding import get_dense_embedding, get_embedding_dim


@pytest.mark.asyncio
async def test_get_dense_embedding_returns_list():
    mock_resp = {"embeddings": [[0.1, 0.2, 0.3]]}
    with patch("app.services.embedding.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json = lambda: mock_resp
        mock_post.return_value.raise_for_status = lambda: None
        vec = await get_dense_embedding("hello")
        assert isinstance(vec, list)
        assert len(vec) == 3
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "qwen3-embedding:0.6b"


@pytest.mark.asyncio
async def test_get_embedding_dim():
    with patch("app.services.embedding.get_dense_embedding", new_callable=AsyncMock) as mock_emb:
        mock_emb.return_value = [0.0] * 512
        dim = await get_embedding_dim()
        assert dim == 512
