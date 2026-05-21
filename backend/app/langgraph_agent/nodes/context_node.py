from langchain.messages import SystemMessage

from app.langgraph_agent.state import AgentState


_SEARCH_INSTRUCTION = """用户要求启用联网搜索。你必须先调用 web_search_tool 工具搜索相关信息，再根据搜索结果回答。

搜索 Query 生成规范（必须遵守）：
1. 专有名词保护：人名、地名、术语必须保持完整，禁止拆成单字搜索。
   错误示例："拉的理论"、"弗的潜意识"、"Chat"
   正确示例："拉康 精神分析 理论"、"弗洛伊德 潜意识"、"ChatGPT"
2. 多关键词组合：用空格分隔 2-4 个关键词，提高搜索精度。
   错误示例："拉康"（太宽泛，可能返回字典页）
   正确示例："拉康 精神分析 镜像阶段 理论"
3. 保留领域词：在专有名词后追加 1-2 个领域关键词。
   示例："拉康" → "拉康 精神分析"；"ChatGPT" → "ChatGPT 人工智能"
4. 语言一致：中文问题用中文 query，英文品牌/技术名词保留英文。

如果搜索返回结果为空或明显不相关（如返回了字典页面），请使用更完整的关键词重新调用 web_search_tool。"""


async def context_node(state: AgentState) -> dict:
    combined_parts = [p for p in [state["memory_context"], state["rag_context"]] if p]

    system_messages = []
    if combined_parts:
        system_content = "\n\n".join(combined_parts)
        system_messages.append(SystemMessage(content=system_content))

    if state["enable_web_search"]:
        system_messages.append(SystemMessage(content=_SEARCH_INSTRUCTION))

    return {"messages": system_messages}
