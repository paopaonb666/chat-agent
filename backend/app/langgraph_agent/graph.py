from typing import Literal

from langgraph.graph import StateGraph, START, END
from langchain.messages import AIMessage

from app.langgraph_agent.state import AgentState
from app.langgraph_agent.nodes import (
    memory_node,
    rag_node,
    context_node,
    llm_node,
    evaluation_node,
)


def _route_after_llm(state: AgentState) -> Literal["search_node", "evaluation_node"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "search_node"
    return "evaluation_node"


async def _search_node(state: AgentState) -> dict:
    """执行网络搜索工具调用并返回工具消息。"""
    from langchain.messages import ToolMessage
    from app.langgraph_agent.tools import web_search_tool

    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    tool_messages = []
    for tc in tool_calls:
        if tc["name"] == "web_search_tool":
            result = await web_search_tool.ainvoke(tc["args"])
            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tc["id"],
            ))

    return {"messages": tool_messages}


def _route_after_eval(state: AgentState) -> Literal["llm_node", "__end__"]:
    if state.get("continue_loop", False):
        return "llm_node"
    return END


def build_agent_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("memory_node", memory_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("context_node", context_node)
    builder.add_node("llm_node", llm_node)
    builder.add_node("search_node", _search_node)
    builder.add_node("evaluation_node", evaluation_node)

    # Fan-out from START to memory + RAG (parallel)
    builder.add_edge(START, "memory_node")
    builder.add_edge(START, "rag_node")

    # Both converge at context_node
    builder.add_edge("memory_node", "context_node")
    builder.add_edge("rag_node", "context_node")

    # Context → LLM
    builder.add_edge("context_node", "llm_node")

    # After LLM: tools or evaluate
    builder.add_conditional_edges(
        "llm_node",
        _route_after_llm,
        {"search_node": "search_node", "evaluation_node": "evaluation_node"},
    )

    # After tools: loop back to LLM
    builder.add_edge("search_node", "llm_node")

    # After evaluation: retry or end
    builder.add_conditional_edges(
        "evaluation_node",
        _route_after_eval,
        {"llm_node": "llm_node", END: END},
    )

    return builder.compile()
