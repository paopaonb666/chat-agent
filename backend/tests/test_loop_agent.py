import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import httpx

from app.services.loop_agent import (
    _resolve_search_query,
    _evaluate_quality,
    _build_correction,
    _has_tool_syntax,
    _accumulate_tool_calls,
    LoopAgent,
)


# ── _resolve_search_query ──────────────────────────────────────────────────


def test_resolve_search_query_parses_valid_json():
    tc = {
        "function": {
            "name": "web_search",
            "arguments": '{"query": "black myth wukong review"}',
        }
    }
    result = _resolve_search_query(tc, "fallback")
    assert result == "black myth wukong review"


def test_resolve_search_query_falls_back_on_short_query():
    tc = {
        "function": {
            "name": "web_search",
            "arguments": '{"query": "ab"}',
        }
    }
    result = _resolve_search_query(tc, "fallback message here")
    assert result == "fallback message here"


def test_resolve_search_query_falls_back_on_json_error():
    tc = {
        "function": {
            "name": "web_search",
            "arguments": "not valid json",
        }
    }
    result = _resolve_search_query(tc, "fallback")
    assert result == "fallback"


def test_resolve_search_query_falls_back_on_empty_query():
    tc = {
        "function": {
            "name": "web_search",
            "arguments": '{"query": ""}',
        }
    }
    result = _resolve_search_query(tc, "fallback")
    assert result == "fallback"


# ── _evaluate_quality ──────────────────────────────────────────────────────


def _fake_eval_response(text: str):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
    }
    return resp


@pytest.mark.asyncio
async def test_evaluate_quality_returns_ok():
    fake = _fake_eval_response("OK")
    with patch("app.services.loop_agent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        is_ok, reason = await _evaluate_quality(
            "what is AI?", "AI is artificial intelligence.", "key", "url", "model"
        )
    assert is_ok is True
    assert reason == ""


@pytest.mark.asyncio
async def test_evaluate_quality_returns_fail():
    fake = _fake_eval_response("FAIL: response is irrelevant to the question")
    with patch("app.services.loop_agent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        is_ok, reason = await _evaluate_quality(
            "user question", "bad answer", "key", "url", "model"
        )
    assert is_ok is False
    assert "irrelevant" in reason


@pytest.mark.asyncio
async def test_evaluate_quality_returns_ok_on_api_error():
    resp = MagicMock()
    resp.status_code = 500
    with patch("app.services.loop_agent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=resp)
        is_ok, reason = await _evaluate_quality(
            "user question", "answer", "key", "url", "model"
        )
    assert is_ok is True


@pytest.mark.asyncio
async def test_evaluate_quality_returns_ok_on_network_error():
    with patch("app.services.loop_agent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        is_ok, reason = await _evaluate_quality(
            "user question", "answer", "key", "url", "model"
        )
    assert is_ok is True


# ── _build_correction ──────────────────────────────────────────────────────


def test_build_correction_includes_reason():
    result = _build_correction("response was too short")
    assert "response was too short" in result
    assert "修正" in result
    assert "web_search" in result


# ── _has_tool_syntax ───────────────────────────────────────────────────────


def test_has_tool_syntax_detects_xml():
    assert _has_tool_syntax("<function_calls>\n<invoke name=\"web_search\">")


def test_has_tool_syntax_detects_function_call_syntax():
    assert _has_tool_syntax('web_search({"query": "test"})')


def test_has_tool_syntax_detects_json_block():
    assert _has_tool_syntax('```json\n{"tool_calls": []}\n```')


def test_has_tool_syntax_returns_false_for_normal_text():
    assert not _has_tool_syntax("这是正常的回复内容，没有任何工具调用语法。")


def test_has_tool_syntax_returns_false_for_plain_english():
    assert not _has_tool_syntax("The weather today is sunny with a high of 25 degrees.")


def test_has_tool_syntax_detects_tool_call_block():
    assert _has_tool_syntax("```tool_calls\nweb_search(query='test')\n```")


# ── _accumulate_tool_calls ─────────────────────────────────────────────────


def test_accumulate_tool_calls_merges_deltas():
    tool_calls_by_index = {}

    # First delta: index 0 with id and function name
    delta1 = [
        {"index": 0, "id": "call_abc", "type": "function",
         "function": {"name": "web_search", "arguments": '{"query":'}},
    ]
    _accumulate_tool_calls(tool_calls_by_index, delta1)

    assert 0 in tool_calls_by_index
    assert tool_calls_by_index[0]["id"] == "call_abc"
    assert tool_calls_by_index[0]["function"]["name"] == "web_search"
    assert tool_calls_by_index[0]["function"]["arguments"] == '{"query":'

    # Second delta: index 0 with more arguments
    delta2 = [
        {"index": 0, "function": {"arguments": '"test"'}},
    ]
    _accumulate_tool_calls(tool_calls_by_index, delta2)
    assert tool_calls_by_index[0]["function"]["arguments"] == '{"query":"test"'

    # Third delta: closing args
    delta3 = [
        {"index": 0, "function": {"arguments": "}"}},
    ]
    _accumulate_tool_calls(tool_calls_by_index, delta3)
    assert tool_calls_by_index[0]["function"]["arguments"] == '{"query":"test"}'


def test_accumulate_tool_calls_multiple_indices():
    tool_calls_by_index = {}

    delta = [
        {"index": 0, "id": "call_0", "type": "function",
         "function": {"name": "web_search", "arguments": '{"query":"q1"}'}},
        {"index": 1, "id": "call_1", "type": "function",
         "function": {"name": "web_search", "arguments": '{"query":"q2"}'}},
    ]
    _accumulate_tool_calls(tool_calls_by_index, delta)

    assert len(tool_calls_by_index) == 2
    assert tool_calls_by_index[0]["id"] == "call_0"
    assert tool_calls_by_index[1]["id"] == "call_1"


# ── Helper to build fake streaming responses ────────────────────────────────


class FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    async def aread(self):
        return b""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeAsyncClient:
    """Mimics httpx.AsyncClient for LoopAgent tests. Supports async context
    manager on both the client itself and its .stream() return value."""

    def __init__(self, lines, should_fail=False):
        self._lines = lines
        self._should_fail = should_fail
        self.last_stream_call = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method, url, **kwargs):
        self.last_stream_call = (method, url, kwargs)
        if self._should_fail:
            raise httpx.HTTPError("connection lost")
        return FakeStreamResponse(self._lines)


def _llm_content_stream(content="test answer", finish_reason="stop"):
    return [
        f'data: {{"choices":[{{"delta":{{"content":"{content}"}},"finish_reason":"{finish_reason}"}}]}}',
        "data: [DONE]",
    ]


def _llm_empty_stream():
    return [
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]


def _llm_tool_call_stream():
    return [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"web_search","arguments":"{\\"query\\":\\"test query\\"}"}}]},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]


# ── LoopAgent iteration tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_agent_exits_on_quality_pass():
    agent = LoopAgent(max_iterations=5)
    messages = [{"role": "user", "content": "hello"}]

    fake_client = FakeAsyncClient(_llm_content_stream("Hi there!"))

    with patch("app.services.loop_agent.httpx.AsyncClient", return_value=fake_client), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")):
        events = []
        async for event in agent.run(
            messages, "hello", False, "key", "http://url", "model"
        ):
            events.append(event)

    assert agent.final_content == "Hi there!"
    step_events = [e for e in events if '"type": "step"' in e]
    assert any("通过" in e for e in step_events)


@pytest.mark.asyncio
async def test_loop_agent_exits_on_max_iterations():
    agent = LoopAgent(max_iterations=2)
    messages = [{"role": "user", "content": "complex question"}]

    call_count = [0]

    def make_fake_client():
        call_count[0] += 1
        return FakeAsyncClient(_llm_content_stream(f"attempt {call_count[0]}"))

    with patch("app.services.loop_agent.httpx.AsyncClient") as mock_client_class, \
         patch("app.services.loop_agent._evaluate_quality", return_value=(False, "not good enough")):
        mock_client_class.side_effect = lambda: make_fake_client()

        events = []
        async for event in agent.run(
            messages, "complex question", False, "key", "http://url", "model"
        ):
            events.append(event)

    step_events = [e for e in events if '"type": "step"' in e]
    assert any("达到最大迭代次数" in e for e in step_events)


@pytest.mark.asyncio
async def test_loop_agent_exits_on_same_failure_limit():
    agent = LoopAgent(max_iterations=10)
    messages = [{"role": "user", "content": "q"}]

    call_count = [0]

    def make_fake_client():
        call_count[0] += 1
        return FakeAsyncClient(_llm_content_stream(f"answer {call_count[0]}"))

    with patch("app.services.loop_agent.httpx.AsyncClient") as mock_client_class, \
         patch("app.services.loop_agent._evaluate_quality", return_value=(False, "same reason")):
        mock_client_class.side_effect = lambda: make_fake_client()

        events = []
        async for event in agent.run(
            messages, "q", False, "key", "http://url", "model"
        ):
            events.append(event)

    step_events = [e for e in events if '"type": "step"' in e]
    assert any("连续" in e and "次相同失败" in e for e in step_events)
    assert call_count[0] == 3


@pytest.mark.asyncio
async def test_loop_agent_handles_tool_call_then_text():
    agent = LoopAgent(max_iterations=5)
    messages = [{"role": "user", "content": "search for something"}]

    client1 = FakeAsyncClient(_llm_tool_call_stream())
    client2 = FakeAsyncClient(_llm_content_stream("Based on search, the answer is 42."))

    mock_web_sources = [{"title": "Result", "url": "https://x.com", "snippet": "snip", "position": 1}]

    with patch("app.services.loop_agent.httpx.AsyncClient") as mock_client_class, \
         patch("app.services.loop_agent.web_search", return_value=mock_web_sources), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")):
        mock_client_class.side_effect = [client1, client2]

        events = []
        async for event in agent.run(
            messages, "search for something", True, "key", "http://url", "model"
        ):
            events.append(event)

    step_events = [e for e in events if '"type": "step"' in e]
    assert any("web_search" in e for e in step_events)
    source_events = [e for e in events if '"type": "sources"' in e]
    assert len(source_events) == 1
    assert agent.final_content == "Based on search, the answer is 42."


@pytest.mark.asyncio
async def test_loop_agent_handles_empty_content():
    agent = LoopAgent(max_iterations=5)
    messages = [{"role": "user", "content": "hello"}]

    fake_client = FakeAsyncClient(_llm_empty_stream())

    with patch("app.services.loop_agent.httpx.AsyncClient", return_value=fake_client):
        events = []
        async for event in agent.run(
            messages, "hello", False, "key", "http://url", "model"
        ):
            events.append(event)

    error_events = [e for e in events if '"type": "step"' in e and '"error"' in e]
    assert len(error_events) >= 1
    assert any("空内容" in e for e in error_events)


@pytest.mark.asyncio
async def test_loop_agent_handles_http_error():
    agent = LoopAgent(max_iterations=3)
    messages = [{"role": "user", "content": "hello"}]

    call_count = [0]

    def make_client():
        call_count[0] += 1
        if call_count[0] == 1:
            return FakeAsyncClient([], should_fail=True)
        return FakeAsyncClient(_llm_content_stream("recovered answer"))

    with patch("app.services.loop_agent.httpx.AsyncClient") as mock_client_class, \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")):
        mock_client_class.side_effect = lambda: make_client()

        events = []
        async for event in agent.run(
            messages, "hello", False, "key", "http://url", "model"
        ):
            events.append(event)

    assert agent.final_content == "recovered answer"
    error_events = [e for e in events if "网络错误" in e]
    assert len(error_events) >= 1
