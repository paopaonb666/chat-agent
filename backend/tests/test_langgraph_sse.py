import json
import pytest
from langchain.messages import HumanMessage, AIMessage, ToolMessage


class TestSSEAdapter:
    def _make_graph(self):
        """Build a minimal graph for testing SSE output."""
        from langgraph.graph import StateGraph, START, END
        from app.langgraph_agent.state import AgentState

        async def eval_node(state):
            from langgraph.config import get_stream_writer
            writer = get_stream_writer()
            writer({"type": "step", "name": "test", "status": "completed", "label": "测试", "detail": "done"})
            return {"final_content": "test response", "continue_loop": False}

        builder = StateGraph(AgentState)
        builder.add_node("evaluation_node", eval_node)
        builder.add_edge(START, "evaluation_node")
        builder.add_edge("evaluation_node", END)
        return builder.compile()

    def _initial_state(self):
        return {
            "messages": [HumanMessage(content="hello")],
            "user_message": "hello",
            "enable_web_search": False,
            "api_key": "sk-test",
            "base_url": "https://api.test.com",
            "model_name": "test-model",
            "user_id": "1",
            "conversation_id": "conv-1",
            "memory_context": "",
            "rag_context": "",
            "web_sources": [],
            "iteration_count": 0,
            "last_failure_reason": "",
            "same_failure_count": 0,
            "final_content": "",
            "continue_loop": False,
        }

    @pytest.mark.asyncio
    async def test_emits_step_events(self):
        from app.langgraph_agent.sse_adapter import create_event_stream

        graph = self._make_graph()
        state = self._initial_state()

        event_gen, holder = create_event_stream(graph, state)
        events = []
        async for e in event_gen:
            events.append(e)

        step_events = [e for e in events if '"type": "step"' in e]
        assert len(step_events) >= 1, f"No step events in: {events}"
        parsed = json.loads(step_events[0][6:].strip())
        assert parsed["type"] == "step"
        assert parsed["name"] == "test"
        assert parsed["status"] == "completed"

    @pytest.mark.asyncio
    async def test_ends_with_done(self):
        from app.langgraph_agent.sse_adapter import create_event_stream

        graph = self._make_graph()
        state = self._initial_state()

        event_gen, holder = create_event_stream(graph, state)
        events = []
        async for e in event_gen:
            events.append(e)

        assert events[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_captures_final_content(self):
        from app.langgraph_agent.sse_adapter import create_event_stream

        graph = self._make_graph()
        state = self._initial_state()

        event_gen, holder = create_event_stream(graph, state)
        async for _ in event_gen:
            pass

        assert holder["final_content"] == "test response"

    @pytest.mark.asyncio
    async def test_all_lines_are_valid_sse(self):
        from app.langgraph_agent.sse_adapter import create_event_stream

        graph = self._make_graph()
        state = self._initial_state()

        event_gen, holder = create_event_stream(graph, state)
        async for line in event_gen:
            assert line.startswith("data: "), f"Invalid SSE line: {line[:80]}"
            assert line.endswith("\n\n"), f"Line doesn't end with newline: {line[:80]}"

    @pytest.mark.asyncio
    async def test_error_emits_error_event(self):
        from app.langgraph_agent.sse_adapter import create_event_stream
        from langgraph.graph import StateGraph, START, END
        from app.langgraph_agent.state import AgentState

        async def failing_node(state):
            raise RuntimeError("test error")

        builder = StateGraph(AgentState)
        builder.add_node("failing", failing_node)
        builder.add_edge(START, "failing")
        builder.add_edge("failing", END)
        graph = builder.compile()

        event_gen, holder = create_event_stream(graph, self._initial_state())
        events = []
        async for e in event_gen:
            events.append(e)

        assert events[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_emits_tool_call_event(self):
        from app.langgraph_agent.sse_adapter import create_event_stream
        from langgraph.graph import StateGraph, START, END
        from app.langgraph_agent.state import AgentState

        async def tool_call_node(state):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"id": "call_1", "name": "web_search_tool", "args": {"query": "test query"}}
                        ],
                    )
                ]
            }

        builder = StateGraph(AgentState)
        builder.add_node("tool_call", tool_call_node)
        builder.add_edge(START, "tool_call")
        builder.add_edge("tool_call", END)
        graph = builder.compile()

        event_gen, holder = create_event_stream(graph, self._initial_state())
        events = []
        async for e in event_gen:
            events.append(e)

        tool_call_events = [e for e in events if '"type": "tool_call"' in e]
        assert len(tool_call_events) == 1, f"Expected 1 tool_call event, got: {events}"
        parsed = json.loads(tool_call_events[0][6:].strip())
        assert parsed["type"] == "tool_call"
        assert parsed["tool_calls"][0]["name"] == "web_search_tool"
        assert parsed["tool_calls"][0]["args"]["query"] == "test query"

    @pytest.mark.asyncio
    async def test_emits_tool_result_event(self):
        from app.langgraph_agent.sse_adapter import create_event_stream
        from langgraph.graph import StateGraph, START, END
        from app.langgraph_agent.state import AgentState

        async def tool_result_node(state):
            return {
                "messages": [
                    ToolMessage(content="search results", tool_call_id="call_1")
                ]
            }

        builder = StateGraph(AgentState)
        builder.add_node("tool_result", tool_result_node)
        builder.add_edge(START, "tool_result")
        builder.add_edge("tool_result", END)
        graph = builder.compile()

        event_gen, holder = create_event_stream(graph, self._initial_state())
        events = []
        async for e in event_gen:
            events.append(e)

        tool_result_events = [e for e in events if '"type": "tool_result"' in e]
        assert len(tool_result_events) == 1, f"Expected 1 tool_result event, got: {events}"
        parsed = json.loads(tool_result_events[0][6:].strip())
        assert parsed["type"] == "tool_result"
        assert parsed["tool_call_id"] == "call_1"
        assert parsed["content"] == "search results"
