import asyncio
import logging

from langgraph.config import get_stream_writer

from app.services.rag_pipeline import run_rag
from app.langgraph_agent.state import AgentState

logger = logging.getLogger(__name__)


async def rag_node(state: AgentState) -> dict:
    writer = get_stream_writer()

    writer({
        "type": "step",
        "name": "rag_retrieval",
        "status": "running",
        "label": "知识库检索",
        "detail": "正在检索历史对话...",
    })

    rag_context = ""
    logger.info("RAG node: query=%.50s conv_id=%s", state["user_message"], state["conversation_id"])
    try:
        messages_for_rag = [
            {"role": getattr(m, "type", "user"), "content": getattr(m, "content", "")}
            for m in state["messages"]
        ]
        raw_uid = state.get("user_id", "")
        user_id_int = int(raw_uid) if raw_uid and str(raw_uid).isdigit() else None
        rag_context = await asyncio.wait_for(
            run_rag(
                None,                     # db — let run_rag create its own session
                state["user_message"],    # query
                state["conversation_id"], # conversation_id
                user_id_int,              # user_id
                messages_for_rag,         # messages
                query_override=state["user_message"],
            ),
            timeout=10.0,
        )
        if rag_context:
            writer({
                "type": "step",
                "name": "rag_retrieval",
                "status": "completed",
                "label": "知识库检索",
                "detail": "检索到相关内容",
            })
        else:
            writer({
                "type": "step",
                "name": "rag_retrieval",
                "status": "completed",
                "label": "知识库检索",
                "detail": "未检索到相关内容",
            })
    except Exception:
        logger.exception("RAG retrieval failed")
        writer({
            "type": "step",
            "name": "rag_retrieval",
            "status": "error",
            "label": "知识库检索",
            "detail": "检索超时",
        })

    return {"rag_context": rag_context}
