import logging

from langgraph.config import get_stream_writer

from app.services.memory_client import get_memory
from app.langgraph_agent.state import AgentState

logger = logging.getLogger(__name__)


async def memory_node(state: AgentState) -> dict:
    writer = get_stream_writer()

    writer({
        "type": "step",
        "name": "memory_check",
        "status": "running",
        "label": "记忆检索",
        "detail": "正在搜索相关记忆...",
    })

    memory_context = ""
    try:
        memory = get_memory()
        results = memory.search(
            query=state["user_message"],
            filters={"user_id": state["user_id"]},
            top_k=3,
        )
        items = results.get("results", []) if isinstance(results, dict) else []
        if items:
            lines = [f"- {r['memory']}" for r in items]
            memory_context = "以下是该用户的长期记忆中存储的相关信息：\n" + "\n".join(lines)
            writer({
                "type": "step",
                "name": "memory_check",
                "status": "completed",
                "label": "记忆检索",
                "detail": f"找到 {len(items)} 条相关记忆",
            })
        else:
            writer({
                "type": "step",
                "name": "memory_check",
                "status": "completed",
                "label": "记忆检索",
                "detail": "未找到相关记忆",
            })
    except Exception:
        logger.exception("Memory search failed")
        writer({
            "type": "step",
            "name": "memory_check",
            "status": "error",
            "label": "记忆检索",
            "detail": "检索失败",
        })

    return {"memory_context": memory_context}
