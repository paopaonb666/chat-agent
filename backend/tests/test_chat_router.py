import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db import Base, get_db
from app import models  # noqa: F401 — registers tables with Base.metadata

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


client = TestClient(app)


@pytest.fixture(scope="function", autouse=True)
def setup_db():
    app.dependency_overrides[get_db] = override_get_db
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


def test_chat_stream():
    resp = client.post("/api/v1/conversations", json={"title": "Stream Test"})
    conv_id = resp.json()["id"]

    fake_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: [DONE]',
    ]

    fake_client = FakeAsyncClient(fake_lines)
    with patch("app.routers.chat.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value=""), \
         patch("app.routers.chat._index_message"):
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "hi",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "Hello" in body
        assert "[DONE]" in body
        # Default model uses DeepSeek endpoint
        assert "api.deepseek.com" in fake_client.last_stream_call[0][1]


def test_chat_stream_with_model():
    resp = client.post("/api/v1/conversations", json={"title": "Zhipu Stream", "model": "glm-4"})
    conv_id = resp.json()["id"]

    fake_lines = [
        'data: {"choices":[{"delta":{"content":"Nihao"}}]}',
        'data: [DONE]',
    ]

    fake_client = FakeAsyncClient(fake_lines)
    with patch("app.routers.chat.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value=""), \
         patch("app.routers.chat._index_message"):
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "hi",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "Nihao" in resp.text
        # Verify Zhipu endpoint is used
        assert "open.bigmodel.cn" in fake_client.last_stream_call[0][1]
        # Verify model name in request body
        assert fake_client.last_stream_call[1]["json"]["model"] == "glm-4"


def test_chat_stream_with_rag_context():
    resp = client.post("/api/v1/conversations", json={"title": "RAG Test"})
    conv_id = resp.json()["id"]

    fake_lines = [
        'data: {"choices":[{"delta":{"content":"Based on history"}}]}',
        'data: [DONE]',
    ]
    fake_client = FakeAsyncClient(fake_lines)
    with patch("app.routers.chat.httpx.AsyncClient", return_value=fake_client), \
         patch("app.routers.chat.run_rag", return_value="历史对话上下文"), \
         patch("app.routers.chat._index_message"):
        resp = client.post("/api/v1/chat/completions", json={
            "conversation_id": conv_id,
            "message": "what did we discuss",
        })
        assert resp.status_code == 200
        request_json = fake_client.last_stream_call[1]["json"]
        assert request_json["messages"][0]["role"] == "system"
        assert "历史对话" in request_json["messages"][0]["content"]
