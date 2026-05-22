import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain.messages import HumanMessage, AIMessage, SystemMessage


def _make_state(**overrides):
    """Helper: build a minimal AgentState dict for node tests."""
    defaults = {
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
    defaults.update(overrides)
    return defaults


def _mock_writer():
    events = []

    def writer(data):
        events.append(data)

    writer.events = events
    return writer


class TestMemoryNode:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_memories(self):
        from app.langgraph_agent.nodes.memory_node import memory_node

        state = _make_state()
        mock_memory = MagicMock()
        mock_memory.search.return_value = {}

        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.memory_node.get_memory", return_value=mock_memory):
            with patch("app.langgraph_agent.nodes.memory_node.get_stream_writer", return_value=writer):
                result = await memory_node(state)

        assert result["memory_context"] == ""

    @pytest.mark.asyncio
    async def test_returns_formatted_memories(self):
        from app.langgraph_agent.nodes.memory_node import memory_node

        state = _make_state()
        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [{"memory": "User likes Python"}, {"memory": "User lives in Beijing"}]
        }

        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.memory_node.get_memory", return_value=mock_memory):
            with patch("app.langgraph_agent.nodes.memory_node.get_stream_writer", return_value=writer):
                result = await memory_node(state)

        assert "Python" in result["memory_context"]
        assert "Beijing" in result["memory_context"]
        assert "严禁编造" in result["memory_context"]

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        from app.langgraph_agent.nodes.memory_node import memory_node

        state = _make_state()
        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.memory_node.get_memory", side_effect=Exception("db down")):
            with patch("app.langgraph_agent.nodes.memory_node.get_stream_writer", return_value=writer):
                result = await memory_node(state)

        assert result["memory_context"] == ""

    @pytest.mark.asyncio
    async def test_emits_step_events(self):
        from app.langgraph_agent.nodes.memory_node import memory_node

        state = _make_state()
        mock_memory = MagicMock()
        mock_memory.search.return_value = {}

        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.memory_node.get_memory", return_value=mock_memory):
            with patch("app.langgraph_agent.nodes.memory_node.get_stream_writer", return_value=writer):
                await memory_node(state)

        assert len(writer.events) == 2  # running + completed
        assert writer.events[0]["type"] == "step"
        assert writer.events[0]["name"] == "memory_check"
        assert writer.events[0]["status"] == "running"


class TestRagNode:
    @pytest.mark.asyncio
    async def test_returns_rag_context(self):
        from app.langgraph_agent.nodes.rag_node import rag_node

        state = _make_state()
        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.rag_node.run_rag", new=AsyncMock(return_value="RAG context here")):
            with patch("app.langgraph_agent.nodes.rag_node.get_stream_writer", return_value=writer):
                result = await rag_node(state)

        assert result["rag_context"] == "RAG context here"

    @pytest.mark.asyncio
    async def test_handles_timeout(self):
        from app.langgraph_agent.nodes.rag_node import rag_node
        import asyncio

        state = _make_state()
        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.rag_node.run_rag", new=AsyncMock(side_effect=asyncio.TimeoutError())):
            with patch("app.langgraph_agent.nodes.rag_node.get_stream_writer", return_value=writer):
                result = await rag_node(state)

        assert result["rag_context"] == ""

    @pytest.mark.asyncio
    async def test_emits_step_events(self):
        from app.langgraph_agent.nodes.rag_node import rag_node

        state = _make_state()
        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.rag_node.run_rag", new=AsyncMock(return_value="ctx")):
            with patch("app.langgraph_agent.nodes.rag_node.get_stream_writer", return_value=writer):
                await rag_node(state)

        assert len(writer.events) == 2
        assert writer.events[0]["name"] == "rag_retrieval"
        assert writer.events[0]["status"] == "running"
        assert writer.events[1]["status"] == "completed"


class TestContextNode:
    @pytest.mark.asyncio
    async def test_no_context_produces_no_system_messages(self):
        from app.langgraph_agent.nodes.context_node import context_node

        state = _make_state(memory_context="", rag_context="")
        result = await context_node(state)

        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_combines_memory_and_rag(self):
        from app.langgraph_agent.nodes.context_node import context_node

        state = _make_state(memory_context="Memory content", rag_context="RAG content")
        result = await context_node(state)

        assert len(result["messages"]) == 2
        # First message: guardrail instruction
        assert "严禁编造" in result["messages"][0].content
        # Second message: combined memory + RAG
        assert "Memory content" in result["messages"][1].content
        assert "RAG content" in result["messages"][1].content

    @pytest.mark.asyncio
    async def test_adds_search_instruction_with_examples_when_enabled(self):
        from app.langgraph_agent.nodes.context_node import context_node

        state = _make_state(enable_web_search=True)
        result = await context_node(state)

        assert len(result["messages"]) == 1
        content = result["messages"][0].content
        assert "联网搜索" in content
        assert "web_search_tool" in content
        assert "专有名词" in content
        assert "错误示例" in content
        assert "正确示例" in content


class TestLLMNode:
    @pytest.mark.asyncio
    async def test_increments_iteration_count(self):
        from app.langgraph_agent.nodes.llm_node import llm_node

        state = _make_state(iteration_count=2)
        mock_response = AIMessage(content="response")
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.langgraph_agent.nodes.llm_node.ChatOpenAI", return_value=mock_model):
            result = await llm_node(state)

        assert result["iteration_count"] == 3
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "response"

    @pytest.mark.asyncio
    async def test_binds_tools_when_web_search_enabled(self):
        from app.langgraph_agent.nodes.llm_node import llm_node

        state = _make_state(enable_web_search=True)
        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_bound.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        mock_model.bind_tools.return_value = mock_bound

        with patch("app.langgraph_agent.nodes.llm_node.ChatOpenAI", return_value=mock_model):
            await llm_node(state)

        mock_model.bind_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_appends_quality_warning_after_bad_tool_result(self):
        from app.langgraph_agent.nodes.llm_node import llm_node
        from langchain.messages import ToolMessage

        state = _make_state(
            messages=[
                HumanMessage(content="hello"),
                ToolMessage(content="[系统提示] 本次搜索的原始关键词被判定为质量不佳，已自动优化", tool_call_id="tc1"),
            ],
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        with patch("app.langgraph_agent.nodes.llm_node.ChatOpenAI", return_value=mock_model):
            await llm_node(state)

        # 调用模型时传入的 messages 应包含追加的系统提示
        called_messages = mock_model.ainvoke.call_args[0][0]
        assert any(isinstance(m, SystemMessage) and "质量不佳" in m.content for m in called_messages)

    @pytest.mark.asyncio
    async def test_appends_no_results_warning_after_empty_tool_result(self):
        from app.langgraph_agent.nodes.llm_node import llm_node
        from langchain.messages import ToolMessage

        state = _make_state(
            messages=[
                HumanMessage(content="hello"),
                ToolMessage(content="未找到相关搜索结果。注意：请检查搜索关键词", tool_call_id="tc1"),
            ],
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        with patch("app.langgraph_agent.nodes.llm_node.ChatOpenAI", return_value=mock_model):
            await llm_node(state)

        called_messages = mock_model.ainvoke.call_args[0][0]
        assert any(isinstance(m, SystemMessage) and "未返回任何结果" in m.content for m in called_messages)


class TestEvaluationNode:
    @pytest.mark.asyncio
    async def test_passes_quality_check(self):
        from app.langgraph_agent.nodes.evaluation_node import evaluation_node

        state = _make_state(
            messages=[HumanMessage(content="hello"), AIMessage(content="Hi there!")],
            iteration_count=1,
        )

        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.evaluation_node._evaluate_quality", new=AsyncMock(return_value=(True, ""))):
            with patch("langgraph.config.get_stream_writer", return_value=writer, side_effect=None):
                result = await evaluation_node(state)

        assert result["final_content"] == "Hi there!"
        assert result["continue_loop"] is False

    @pytest.mark.asyncio
    async def test_fails_and_retries(self):
        from app.langgraph_agent.nodes.evaluation_node import evaluation_node

        state = _make_state(
            messages=[HumanMessage(content="hello"), AIMessage(content="bad response")],
            iteration_count=1,
        )

        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.evaluation_node._evaluate_quality", new=AsyncMock(return_value=(False, "回答不相关"))):
            with patch("langgraph.config.get_stream_writer", return_value=writer):
                result = await evaluation_node(state)

        assert result["continue_loop"] is True
        assert len(result["messages"]) == 1  # correction message
        assert "不合格" in result["messages"][0].content
        assert result["same_failure_count"] == 1

    @pytest.mark.asyncio
    async def test_stops_on_max_iterations(self):
        from app.langgraph_agent.nodes.evaluation_node import evaluation_node

        state = _make_state(
            messages=[HumanMessage(content="hello"), AIMessage(content="bad")],
            iteration_count=50,
        )

        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.evaluation_node._evaluate_quality", new=AsyncMock(return_value=(False, "bad"))):
            with patch("langgraph.config.get_stream_writer", return_value=writer):
                result = await evaluation_node(state)

        assert result["continue_loop"] is False

    @pytest.mark.asyncio
    async def test_stops_on_repeated_failure(self):
        from app.langgraph_agent.nodes.evaluation_node import evaluation_node

        state = _make_state(
            messages=[HumanMessage(content="hello"), AIMessage(content="bad")],
            iteration_count=5,
            same_failure_count=2,
            last_failure_reason="bad",
        )

        writer = _mock_writer()
        with patch("app.langgraph_agent.nodes.evaluation_node._evaluate_quality", new=AsyncMock(return_value=(False, "bad"))):
            with patch("langgraph.config.get_stream_writer", return_value=writer):
                result = await evaluation_node(state)

        assert result["continue_loop"] is False  # 3rd same failure → stop

    @pytest.mark.asyncio
    async def test_handles_empty_content(self):
        from app.langgraph_agent.nodes.evaluation_node import evaluation_node

        state = _make_state(
            messages=[HumanMessage(content="hello"), AIMessage(content="")],
            iteration_count=1,
        )

        writer = _mock_writer()
        with patch("langgraph.config.get_stream_writer", return_value=writer):
            result = await evaluation_node(state)

        assert result["final_content"] == ""
        assert result["continue_loop"] is False

    @pytest.mark.asyncio
    async def test_correction_includes_search_query_guidelines(self):
        from app.langgraph_agent.nodes.evaluation_node import evaluation_node

        state = _make_state(
            messages=[HumanMessage(content="hello"), AIMessage(content="bad response")],
            iteration_count=1,
        )

        with patch("app.langgraph_agent.nodes.evaluation_node._evaluate_quality", new=AsyncMock(return_value=(False, "回答不相关"))):
            result = await evaluation_node(state)

        assert result["continue_loop"] is True
        correction = result["messages"][0].content
        assert "错误：" in correction
        assert "正确：" in correction
        assert "web_search_tool" in correction
        assert "字典页" in correction
