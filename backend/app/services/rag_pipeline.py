from app.services.intent import recognize_intent
from app.services.embedding import get_dense_embedding
from app.services.hybrid_search import hybrid_search
from app.services.rerank import rerank_passages


async def run_rag(
    query: str,
    conversation_id: str,
    user_id: int | None,
    messages: list[dict],
    top_k_hybrid: int = 15,
    top_n_rerank: int = 5,
) -> str:
    intent = await recognize_intent(query, messages)
    if not intent.needs_retrieval:
        return ""
    refined = intent.refined_query or query
    dense_vector = await get_dense_embedding(refined)
    hybrid_results = await hybrid_search(refined, dense_vector, user_id, conversation_id, top_k=top_k_hybrid)
    if not hybrid_results:
        return ""
    reranked = await rerank_passages(refined, hybrid_results, top_n=top_n_rerank)
    context_lines = [f"[{i+1}] {r['content']}" for i, r in enumerate(reranked)]
    context = "\n".join(context_lines)
    return f"以下是从历史对话中检索到的相关信息：\n{context}\n\n请基于以上信息回答用户问题。"
