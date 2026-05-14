import jieba
from rank_bm25 import BM25Okapi
from app.services.milvus_store import search_dense as milvus_search_dense, get_milvus_client
from app.db import SessionLocal
from app.models import Message, Conversation

_milvus_client = None


def _get_client():
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = get_milvus_client()
    return _milvus_client


def _tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))


def _bm25_search(query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
    if not documents:
        return []
    tokenized_docs = [_tokenize(d["content"]) for d in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [documents[i] for i, _ in ranked]


def _fetch_candidates_from_pg(user_id: int | None, limit: int = 200) -> list[dict]:
    db = SessionLocal()
    try:
        q = db.query(Message).join(Conversation).order_by(Message.id.desc())
        if user_id is not None:
            q = q.filter(Conversation.user_id == user_id)
        rows = q.limit(limit).all()
        return [
            {
                "id": r.id,
                "content": r.content,
                "role": r.role,
                "conversation_id": r.conversation_id,
                "message_id": r.id,
            }
            for r in rows
        ]
    finally:
        db.close()


def reciprocal_rank_fusion(list1: list[dict], list2: list[dict], k: int = 60) -> list[dict]:
    scores: dict = {}
    items: dict = {}
    for rank, item in enumerate(list1):
        key = item.get("id") or item.get("message_id")
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        items[key] = item
    for rank, item in enumerate(list2):
        key = item.get("id") or item.get("message_id")
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        items[key] = item
    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    seen_ids = set()
    results = []
    for key in ranked_keys:
        item = items[key]
        item_id = id(item)
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            results.append(item)
    return results


async def hybrid_search(
    query: str,
    dense_embedding: list[float],
    user_id: int | None,
    conversation_id: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    client = _get_client()
    dense_results = milvus_search_dense(client, dense_embedding, user_id, top_k=top_k * 2)
    candidates = _fetch_candidates_from_pg(user_id, limit=200)
    bm25_results = _bm25_search(query, candidates, top_k=top_k * 2)
    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)
    return fused[:top_k]
