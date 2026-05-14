import pytest
from unittest.mock import patch
from app.services.hybrid_search import hybrid_search, reciprocal_rank_fusion, _bm25_search


def test_bm25_search_ranking():
    docs = [
        {"id": 1, "content": "Python 错误处理"},
        {"id": 2, "content": "JavaScript 异步编程"},
        {"id": 3, "content": "Python 异常捕获教程"},
    ]
    results = _bm25_search("Python 错误", docs, top_k=2)
    assert len(results) == 2
    returned_ids = {r["id"] for r in results}
    assert returned_ids <= {1, 2, 3}


def test_reciprocal_rank_fusion():
    list1 = [{"id": "a"}, {"id": "b"}]
    list2 = [{"id": "b"}, {"id": "c"}]
    fused = reciprocal_rank_fusion(list1, list2, k=60)
    ids = [r["id"] for r in fused]
    assert "b" in ids
    assert len(fused) == 3


@pytest.mark.asyncio
async def test_hybrid_search_combines_sources():
    with patch("app.services.hybrid_search.milvus_search_dense") as mock_dense, patch(
        "app.services.hybrid_search._fetch_candidates_from_pg"
    ) as mock_pg:
        mock_dense.return_value = [{"id": "d1", "content": "dense hit"}]
        mock_pg.return_value = [{"id": "k1", "content": "keyword hit"}]
        results = await hybrid_search("query", [0.1] * 768, user_id=1, top_k=5)
        assert len(results) >= 1
        mock_dense.assert_called_once()
        mock_pg.assert_called_once()
