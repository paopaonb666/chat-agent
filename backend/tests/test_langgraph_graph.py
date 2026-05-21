import pytest
from langchain.messages import HumanMessage, AIMessage


class TestGraphConstruction:
    def test_graph_compiles(self):
        from app.langgraph_agent.graph import build_agent_graph

        graph = build_agent_graph()
        assert graph is not None
        assert "messages" in graph.channels
        assert "final_content" in graph.channels

    def test_route_after_llm_to_search_when_tool_calls(self):
        from app.langgraph_agent.graph import _route_after_llm

        msg = AIMessage(
            content="",
            tool_calls=[{"name": "web_search_tool", "args": {}, "id": "1"}],
        )
        state = {"messages": [msg]}
        assert _route_after_llm(state) == "search_node"

    def test_route_after_llm_to_evaluation_when_no_tool_calls(self):
        from app.langgraph_agent.graph import _route_after_llm

        msg = AIMessage(content="plain text response")
        state = {"messages": [msg]}
        assert _route_after_llm(state) == "evaluation_node"

    def test_route_after_eval_retry_when_continue(self):
        from app.langgraph_agent.graph import _route_after_eval

        state = {"continue_loop": True}
        assert _route_after_eval(state) == "llm_node"

    def test_route_after_eval_end_when_done(self):
        from app.langgraph_agent.graph import _route_after_eval

        state = {"continue_loop": False}
        from langgraph.graph import END
        assert _route_after_eval(state) == END
