import logging

from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, SystemMessage, ToolMessage

from app.langgraph_agent.state import AgentState
from app.langgraph_agent.tools import web_search_tool

logger = logging.getLogger(__name__)


async def llm_node(state: AgentState) -> dict:
    model = ChatOpenAI(
        model=state["model_name"],
        api_key=state["api_key"],
        base_url=state["base_url"],
        temperature=0.7,
        streaming=True,
    )

    if state["enable_web_search"]:
        model_with_tools = model.bind_tools([web_search_tool])
    else:
        model_with_tools = model

    messages = list(state["messages"])

    # 如果上一条是工具结果且包含质量警告，追加提示
    if messages and isinstance(messages[-1], ToolMessage):
        tool_content = getattr(messages[-1], "content", "") or ""
        if "原始关键词被判定为质量不佳" in tool_content:
            messages.append(SystemMessage(
                content="注意：上一轮搜索的关键词质量不佳，已被系统自动修正。"
                "如果当前搜索结果仍不相关，请使用更完整、更精确的关键词重新搜索。"
            ))
        elif "未找到相关搜索结果" in tool_content:
            messages.append(SystemMessage(
                content="注意：上一轮搜索未返回任何结果。"
                "请检查搜索关键词是否过于狭窄或包含被拆分的专有名词，"
                "使用更完整的关键词重新调用 web_search_tool。"
            ))

    response: AIMessage = await model_with_tools.ainvoke(messages)

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
