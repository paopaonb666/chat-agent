import json
import logging
from typing import AsyncGenerator, Any

from langchain.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph

from app.langgraph_agent.state import AgentState

logger = logging.getLogger(__name__)


def create_event_stream(
    graph: StateGraph,
    initial_state: dict,
) -> tuple[AsyncGenerator[str, None], dict]:
    """从LangGraph执行创建SSE事件流。

    返回 (async_generator, holder)，其中 holder["final_content"] 在生成器耗尽后被填充。
    """
    holder: dict[str, Any] = {"final_content": ""}

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for stream_mode, chunk in graph.astream(
                initial_state,
                stream_mode=["messages", "custom", "updates"],
            ):
                if stream_mode == "messages":
                    message_chunk, metadata = chunk
                    if isinstance(message_chunk, AIMessage) and message_chunk.tool_calls:
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool_calls': message_chunk.tool_calls}, ensure_ascii=False)}\n\n"
                    elif isinstance(message_chunk, ToolMessage):
                        yield f"data: {json.dumps({'type': 'tool_result', 'tool_call_id': message_chunk.tool_call_id, 'content': message_chunk.content}, ensure_ascii=False)}\n\n"
                    elif metadata.get("langgraph_node") == "llm_node":
                        content = getattr(message_chunk, "content", "")
                        if content:
                            yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

                elif stream_mode == "custom":
                    data = chunk
                    event_type = data.get("type", "")
                    if event_type in ("step", "sources"):
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                elif stream_mode == "updates":
                    for node_name, node_output in chunk.items():
                        if node_name == "evaluation_node" and "final_content" in node_output:
                            holder["final_content"] = node_output["final_content"]

        except Exception:
            logger.exception("LangGraph stream error")
            yield f"data: {json.dumps({'error': 'Agent stream error'})}\n\n"

        yield "data: [DONE]\n\n"

    return event_stream(), holder
