from typing import AsyncGenerator

from langgraph.graph import StateGraph

from app.langgraph_agent.graph import build_agent_graph
from app.langgraph_agent.sse_adapter import create_event_stream


class LangGraphAgent:
    """基于LangGraph的聊天代理 — LoopAgent的即插即用替代方案。"""

    def __init__(self):
        self._graph: StateGraph | None = None

    @property
    def graph(self) -> StateGraph:
        if self._graph is None:
            self._graph = build_agent_graph()
        return self._graph

    async def run(
        self,
        initial_state: dict,
    ) -> tuple[AsyncGenerator[str, None], dict]:
        """使用给定的初始状态运行代理。

        返回 (event_stream, holder)，其中：
        - event_stream 生成 SSE 格式的字符串
        - holder["final_content"] 包含流式传输完成后的最终响应
        """
        return create_event_stream(self.graph, initial_state)
