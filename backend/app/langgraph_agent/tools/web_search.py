import asyncio
import logging

from langchain.tools import tool
from langgraph.config import get_stream_writer

from app.services.web_search import web_search, format_web_context
from app.services.query_validator import enhance_query

logger = logging.getLogger(__name__)

SEARCH_TIMEOUT = 8.0


@tool
async def web_search_tool(query: str) -> str:
    """搜索互联网获取最新信息。当需要了解实时新闻、当前事件、
    最新数据或不确定的事实性知识时调用此工具。

    Query 生成规范：
    1. 专有名词保护 —— 人名、地名、术语必须保持完整，禁止拆成单字。
       错误示例："拉的理论"（"拉康"被拆分）
       正确示例："拉康 精神分析 理论"
    2. 多关键词组合 —— 用空格分隔多个关键词，提高搜索精度。
       错误示例："拉康"（太宽泛）
       正确示例："拉康 精神分析 镜像阶段"
    3. 保留原文语言 —— 中文问题用中文 query，英文术语保留英文。

    Args:
        query: 搜索关键词或问题，使用中文或英文，尽量精简准确
    """
    writer = get_stream_writer()

    # 前置校验：即使 LLM 生成了坏 query，也在工具层拦截并修正
    validation = await enhance_query(query)
    search_query = validation.query

    if validation.was_enhanced:
        writer({
            "type": "step",
            "name": "web_search",
            "status": "running",
            "label": "联网搜索",
            "detail": f"Agent 请求搜索：{query[:40]}...（已优化为：{search_query[:40]}）",
        })
        logger.info("web_search_tool: query enhanced '%s' -> '%s'", query, search_query)
    else:
        writer({
            "type": "step",
            "name": "web_search",
            "status": "running",
            "label": "联网搜索",
            "detail": f"Agent 请求搜索：{query[:40]}...",
        })

    try:
        sources = await asyncio.wait_for(web_search(search_query), timeout=SEARCH_TIMEOUT)
    except Exception:
        logger.exception("web_search_tool failed")
        sources = []

    writer({
        "type": "step",
        "name": "web_search",
        "status": "completed",
        "label": "联网搜索",
        "detail": f"找到 {len(sources)} 条结果",
    })

    writer({
        "type": "sources",
        "sources": sources,
    })

    context = format_web_context(sources)
    if not context:
        return (
            "未找到相关搜索结果。"
            "注意：请检查搜索关键词是否准确，专有名词（如人名、术语）必须保持完整，"
            "不要使用单字或拆分后的片段进行搜索。"
        )

    # 如果 query 被工具层修正过，在返回给 LLM 的上下文中附加提示
    if validation.was_enhanced:
        context += (
            "\n\n[系统提示] 本次搜索的原始关键词被判定为质量不佳，"
            f"已自动优化为：'{search_query}'。"
            "如果搜索结果仍不理想，请使用更完整的关键词重新调用 web_search_tool。"
        )

    return context
