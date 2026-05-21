import logging
import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from app.services.milvus_store import search_dense as milvus_search_dense
from app.models import Message, Conversation

logger = logging.getLogger(__name__)


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


def _fetch_candidates_from_pg(db: Session, user_id: int | None, limit: int = 200) -> list[dict]:
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


def _time_decay_score(
    msg_id: int | str,
    newest_id: int | str,
    max_bonus: float | None = None,
    window_size: int | None = None,
) -> float:
    """Newer messages get a bonus, decaying linearly to 0 for the oldest."""
    if max_bonus is None:
        from app.core.config import settings
        max_bonus = settings.time_decay_max_bonus
    if window_size is None:
        from app.core.config import settings
        window_size = settings.time_decay_window_size

    try:
        newest = int(newest_id)
        msg = int(msg_id)
    except (ValueError, TypeError):
        return 0.0
    if newest <= 1:
        return 0.0
    # Position within the window: newest = 1, oldest in window = 0
    window_start = max(1, newest - window_size)
    position = (msg - window_start) / max(1, newest - window_start)
    position = max(0.0, min(1.0, position))
    return max_bonus * position


def _reciprocal_rank_fusion(
    dense_list: list[dict],
    bm25_list: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict = {}
    items: dict = {}
    newest_id = max(
        (item.get("message_id", item.get("id", 0)) for item in [*dense_list, *bm25_list]),
        default=0,
    )

    def add_contributions(source: list[dict]):
        for rank, item in enumerate(source):
            key = item.get("id") or item.get("message_id")
            rrf = 1.0 / (k + rank + 1.0)
            msg_id = item.get("message_id", item.get("id", 0))
            time_bonus = _time_decay_score(msg_id, newest_id)
            scores[key] = scores.get(key, 0.0) + rrf + time_bonus
            items[key] = item

    add_contributions(dense_list)
    add_contributions(bm25_list)

    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    seen: set[int] = set()
    results: list[dict] = []
    for key in ranked_keys:
        item_id = id(items[key])
        if item_id not in seen:
            seen.add(item_id)
            results.append(items[key])
    return results


async def hybrid_search(
    db: Session,
    query: str,
    dense_embedding: list[float],
    user_id: int | None,
    conversation_id: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    from app.core.milvus import get_milvus_client
    client = get_milvus_client()

    # Step 1: dense semantic search
    dense_results = milvus_search_dense(client, dense_embedding, user_id, top_k=top_k * 2)

    # Step 2: fetch candidates from PostgreSQL
    candidates = _fetch_candidates_from_pg(db, user_id, limit=200)

    # Step 3: semantic pre-filter — only BM25 on dense-matched + recent messages
    dense_ids = {r.get("message_id", r.get("id")) for r in dense_results}
    filtered = [c for c in candidates if c.get("message_id", c.get("id")) in dense_ids]
    # Always keep the most recent batch as fallback
    seen_ids = {c.get("message_id", c.get("id")) for c in filtered}
    for c in candidates[:50]:
        cid = c.get("message_id", c.get("id"))
        if cid not in seen_ids:
            seen_ids.add(cid)
            filtered.append(c)

    # Step 4: BM25 on pre-filtered candidates
    bm25_results = _bm25_search(query, filtered, top_k=top_k * 2)

    # Step 5: fusion with time decay
    fused = _reciprocal_rank_fusion(dense_results, bm25_results, k=60)
    return fused[:top_k]
