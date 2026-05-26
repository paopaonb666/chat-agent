import logging
from sqlalchemy.orm import Session
from app.core.milvus import get_milvus_client
from app.models import KnowledgeDocument
from app.services.embedding import get_dense_embedding
from app.services.knowledge_milvus import search_knowledge_base

logger = logging.getLogger(__name__)


async def search_knowledge_base_rag(db: Session, query: str, user_id: int | None, top_k: int = 5) -> str:
    """Search knowledge base and return formatted context string. Empty string if no results."""
    try:
        dense_vector = await get_dense_embedding(query)
        client = get_milvus_client()

        # Build visible document IDs
        q = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.status == "completed",
            KnowledgeDocument.vector_indexed.is_(True),
        )
        if user_id is not None:
            from app.models import User
            user = db.query(User).filter(User.id == user_id).first()
            if user and user.role != "admin":
                q = q.filter(KnowledgeDocument.visibility.in_(["public", "shared"]))
        visible_docs = q.all()
        if not visible_docs:
            return ""
        document_ids = [d.id for d in visible_docs]

        results = search_knowledge_base(client, dense_vector, document_ids, top_k=top_k)
        if not results:
            return ""

        context_lines = []
        for i, r in enumerate(results):
            line = f"[{i + 1}] {r['content']}"
            title_path = r.get("title_path")
            if title_path:
                line += f" (来源: {title_path})"
            context_lines.append(line)
        context = "\n".join(context_lines)
        return f"以下是从企业文档知识库中检索到的相关信息：\n{context}\n\n请基于以上信息回答用户问题。"
    except Exception:
        logger.exception("Knowledge base search failed")
        return ""
