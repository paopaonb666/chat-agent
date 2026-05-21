import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db import Base, get_db
from app.deps import get_current_user, oauth2_scheme
from app.models import User
from app.core.security import get_password_hash
from app import models  # noqa: F401

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


def override_get_current_user():
    db = TestingSessionLocal()
    try:
        user = db.query(User).filter(User.username == "testuser").first()
        if not user:
            user = User(username="testuser", password_hash=get_password_hash("testpass"), role="admin")
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    finally:
        db.close()


def override_oauth2():
    return "test-token"


client = TestClient(app)


@pytest.fixture(scope="function", autouse=True)
def setup_db():
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[oauth2_scheme] = override_oauth2
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


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
    def __init__(self, lines):
        self._lines = lines
        self.last_stream_call = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, *args, **kwargs):
        self.last_stream_call = (args, kwargs)
        return FakeStreamResponse(self._lines)


def _normal_llm_stream(content="Hello", finish_reason="stop"):
    return [
        f'data: {{"choices":[{{"delta":{{"content":"{content}"}},"finish_reason":"{finish_reason}"}}]}}',
        'data: [DONE]',
    ]


def _tool_call_stream(query="test query"):
    """Simulate LLM response with a web_search tool call."""
    return [
        f'data: {{"choices":[{{"delta":{{"content":"Let me search","tool_calls":[{{"index":0,"id":"call_001","type":"function","function":{{"name":"web_search","arguments":"{{\\"query\\":\\"{query}\\"}}"}}}}]}},"finish_reason":"tool_calls"}}]}}',
        'data: [DONE]',
    ]


def _tool_then_text_stream(query="test query", content="search result based"):
    """LLM first calls tool, then returns text."""
    return (
        _tool_call_stream(query),
        [
            f'data: {{"choices":[{{"delta":{{"content":"{content}"}},"finish_reason":"stop"}}]}}',
            'data: [DONE]',
        ],
    )


# ── CRUD tests ────────────────────────────────────────────────────────────


def test_update_conversation_title():
    resp = client.post("/api/v1/conversations", json={"title": "旧标题"})
    conv_id = resp.json()["id"]
    resp = client.patch(f"/api/v1/conversations/{conv_id}", json={"title": "新标题"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "新标题"
    resp = client.get(f"/api/v1/conversations/{conv_id}")
    assert resp.json()["title"] == "新标题"


def test_update_conversation_not_found():
    resp = client.patch("/api/v1/conversations/nonexistent", json={"title": "x"})
    assert resp.status_code == 404


def test_create_and_list_conversations():
    resp = client.post("/api/v1/conversations", json={"title": "Test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Test"
    assert data["model"] == "deepseek-chat"
    assert "id" in data
    resp = client.get("/api/v1/conversations")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_create_conversation_with_model():
    resp = client.post("/api/v1/conversations", json={"title": "Zhipu", "model": "glm-4"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "glm-4"


def test_get_conversation():
    resp = client.post("/api/v1/conversations", json={"title": "Get Test"})
    conv_id = resp.json()["id"]
    resp = client.get(f"/api/v1/conversations/{conv_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == conv_id


def test_get_conversation_not_found():
    resp = client.get("/api/v1/conversations/nonexistent")
    assert resp.status_code == 404


# ── Streaming tests ───────────────────────────────────────────────────────


def test_chat_stream_auto_title():
    resp = client.post("/api/v1/conversations", json={"title": "新对话"})
    conv_id = resp.json()["id"]

    fake_client = FakeAsyncClient(_normal_llm_stream("你好！我是AI助手"))
    with patch("app.services.loop_agent.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value=""), \
         patch("app.routers.chat.index_message"), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")), \
         patch("app.routers.chat.auto_title_on_first_exchange") as mock_title_gen:
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "你好我叫张三",
        })
        assert resp.status_code == 200

    mock_title_gen.assert_called_once()
    args, _ = mock_title_gen.call_args
    assert args[0] == conv_id
    assert "你好" in args[1]


def test_chat_stream():
    resp = client.post("/api/v1/conversations", json={"title": "Stream Test"})
    conv_id = resp.json()["id"]

    fake_client = FakeAsyncClient(_normal_llm_stream("Hello"))
    with patch("app.services.loop_agent.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value=""), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")), \
         patch("app.routers.chat.index_message") as mock_index:
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "hi",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "Hello" in body
        assert "[DONE]" in body
        assert "api.deepseek.com" in fake_client.last_stream_call[0][1]
    assert mock_index.call_count == 2


def test_chat_stream_with_model():
    resp = client.post("/api/v1/conversations", json={"title": "Zhipu Stream", "model": "glm-4"})
    conv_id = resp.json()["id"]

    fake_client = FakeAsyncClient(_normal_llm_stream("Nihao"))
    with patch("app.services.loop_agent.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value=""), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")), \
         patch("app.routers.chat.index_message"):
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "hi",
        })
        assert resp.status_code == 200
        assert "Nihao" in resp.text
        assert "open.bigmodel.cn" in fake_client.last_stream_call[0][1]
        assert fake_client.last_stream_call[1]["json"]["model"] == "glm-4"


def test_chat_stream_with_rag():
    """RAG context is included in system message."""
    resp = client.post("/api/v1/conversations", json={"title": "RAG Test"})
    conv_id = resp.json()["id"]

    fake_client = FakeAsyncClient(_normal_llm_stream("combined"))
    with patch("app.services.loop_agent.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value="历史对话上下文"), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")), \
         patch("app.routers.chat.index_message"):
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "what did we discuss",
        })
        assert resp.status_code == 200
        request_json = fake_client.last_stream_call[1]["json"]
        assert request_json["messages"][0]["role"] == "system"
        assert "历史对话上下文" in request_json["messages"][0]["content"]


def test_chat_stream_with_web_search_tool():
    """When enable_web_search=true, LLM calls web_search tool and gets results."""
    resp = client.post("/api/v1/conversations", json={"title": "Tool Test"})
    conv_id = resp.json()["id"]

    mock_web_sources = [{"title": "Result1", "url": "https://x.com", "snippet": "snip", "position": 1}]

    # Simulate two LLM calls: first with tool_call, second with text
    with patch("app.services.loop_agent.httpx.AsyncClient") as mock_client_class, \
         patch("app.routers.chat.run_rag", return_value=""), \
         patch("app.routers.chat.index_message"), \
         patch("app.services.loop_agent.web_search", return_value=mock_web_sources), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")):

        # Round 1: returns tool call
        client1 = MagicMock()
        client1.__aenter__.return_value = client1
        client1.__aexit__.return_value = None
        client1.stream.return_value = FakeStreamResponse(_tool_call_stream("black myth wukong review"))
        client1.stream.return_value.status_code = 200

        # Round 2: returns text
        client2 = MagicMock()
        client2.__aenter__.return_value = client2
        client2.__aexit__.return_value = None
        client2.stream.return_value = FakeStreamResponse(_normal_llm_stream("Based on search results"))
        client2.stream.return_value.status_code = 200

        mock_client_class.side_effect = [client1, client2]

        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "black myth wukong review",
            "enable_web_search": True,
        })
        assert resp.status_code == 200
        # Should contain step event for web_search
        assert '"type": "step"' in resp.text
        assert '"name": "web_search"' in resp.text
        # Should contain sources
        assert '"type": "sources"' in resp.text
        assert '"Result1"' in resp.text
        # Should contain final text from second round
        assert "Based on search results" in resp.text


def test_chat_stream_without_web_search():
    """When enable_web_search=false, no tools are sent to LLM."""
    resp = client.post("/api/v1/conversations", json={"title": "No Tool"})
    conv_id = resp.json()["id"]

    fake_client = FakeAsyncClient(_normal_llm_stream("Hello"))
    with patch("app.services.loop_agent.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value=""), \
         patch("app.routers.chat.index_message"), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")):
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "hi",
            "enable_web_search": False,
        })
        assert resp.status_code == 200
        # Verify tools were not in the request
        request_json = fake_client.last_stream_call[1]["json"]
        assert request_json.get("tools") is None


def test_chat_stream_auto_extracts_memory():
    resp = client.post("/api/v1/conversations", json={"title": "Memory Test"})
    conv_id = resp.json()["id"]

    fake_client = FakeAsyncClient(_normal_llm_stream("很高兴认识你"))
    mock_memory = MagicMock()
    with patch("app.services.loop_agent.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value=""), \
         patch("app.routers.chat.index_message"), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")), \
         patch("app.routers.chat.should_extract_memory", return_value=(True, "用户叫炮炮")), \
         patch("app.routers.chat.get_memory", return_value=mock_memory):
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "我叫炮炮",
        })
        assert resp.status_code == 200
        mock_memory.add.assert_called_once()
        # Verify the filtered content was passed, not the raw conversation
        call_args = mock_memory.add.call_args
        assert "炮炮" in str(call_args)


def test_chat_stream_skips_memory_when_no_assistant_response():
    resp = client.post("/api/v1/conversations", json={"title": "No Memory"})
    conv_id = resp.json()["id"]

    fake_client = FakeAsyncClient(['data: [DONE]'])
    mock_memory = MagicMock()
    with patch("app.services.loop_agent.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value=""), \
         patch("app.routers.chat.index_message"), \
         patch("app.services.loop_agent._evaluate_quality", return_value=(True, "")), \
         patch("app.routers.chat.get_memory", return_value=mock_memory):
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "你好",
        })
        assert resp.status_code == 200
        mock_memory.add.assert_not_called()
