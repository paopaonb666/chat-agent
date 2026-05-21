import time
from sqlalchemy.orm import Session
from app.services.embedding import get_dense_embedding
from app.services.hybrid_search import hybrid_search
from app.services.rerank import rerank_passages
from app.core.metrics import (
    get_rag_retrieval_counter,
    get_rag_retrieval_latency,
    get_rag_result_count,
)


async def run_rag(
    db: Session | None,
    query: str,
    conversation_id: str,
    user_id: int | None,
    messages: list[dict],
    *,
    query_override: str = "",
    top_k_hybrid: int = 15,
    top_n_rerank: int = 5,
    min_results_for_rerank: int = 3,
) -> str:
    own_session = False
    if db is None:
        from app.db import SessionLocal
        db = SessionLocal()
        own_session = True
    try:
        refined = query_override or query
        dense_vector = await get_dense_embedding(refined)

        t0 = time.time()
        hybrid_results = await hybrid_search(db, refined, dense_vector, user_id, conversation_id, top_k=top_k_hybrid)
        hybrid_latency = time.time() - t0
        get_rag_retrieval_latency().labels(status="success").observe(hybrid_latency)
        get_rag_result_count().labels(stage="hybrid").observe(len(hybrid_results))

        if not hybrid_results:
            get_rag_retrieval_counter().labels(status="empty").inc()
            return ""

        # Skip rerank when too few results to avoid pointless LLM call
        if len(hybrid_results) < min_results_for_rerank:
            reranked = hybrid_results
        else:
            reranked = await rerank_passages(refined, hybrid_results, top_n=top_n_rerank)
            get_rag_result_count().labels(stage="rerank").observe(len(reranked))

        get_rag_retrieval_counter().labels(status="success").inc()
        context_lines = [f"[{i+1}] {r['content']}" for i, r in enumerate(reranked)]
        context = "\n".join(context_lines)
        return f"以下是从历史对话中检索到的相关信息：\n{context}\n\n请基于以上信息回答用户问题。"
    finally:
        if own_session:
            db.close()
