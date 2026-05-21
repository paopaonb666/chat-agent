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


@patch("app.routers.memory.get_memory")
def test_memory_store_step_events(mock_get_memory):
    """Store endpoint returns SSE step events and calls memory.add()."""
    mock_memory = MagicMock()
    mock_memory.add.return_value = {"results": [{"id": "mem-abc", "memory": "我的名字叫炮炮"}]}
    mock_get_memory.return_value = mock_memory

    conv_resp = client.post("/api/v1/conversations", json={"title": "test", "model": "deepseek-chat"})
    conv_id = conv_resp.json()["id"]

    resp = client.post("/api/v1/memory/store", json={
        "content": "我的名字叫炮炮",
        "conversation_id": conv_id,
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = []
    for line in resp.text.strip().split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    step_events = [e for e in events if e.get("type") == "step"]
    step_names = [s["step"] for s in step_events if s["status"] == "running"]
    assert step_names == ["prepare", "embed", "dedup", "save", "index"]

    step_completed = [s["step"] for s in step_events if s["status"] == "completed"]
    assert step_completed == ["prepare", "embed", "dedup", "save", "index"]

    done_events = [e for e in events if e.get("type") == "done"]
    assert len(done_events) == 1

    # Verify mem0 add() was called with the content
    mock_memory.add.assert_called_once()
    call_args = mock_memory.add.call_args
    messages = call_args[0][0]
    assert messages[0]["content"] == "我的名字叫炮炮"


@patch("app.routers.memory.get_memory")
def test_create_memory_manual(mock_get_memory):
    """POST /memories creates a manual memory via mem0."""
    mock_memory = MagicMock()
    mock_memory.add.return_value = {
        "results": [{"id": "mem-001", "memory": "我喜欢Python编程"}]
    }
    mock_get_memory.return_value = mock_memory

    resp = client.post("/api/v1/memories", json={"content": "我喜欢Python编程"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "我喜欢Python编程"
    assert data["source"] == "manual"
    assert data["id"] == "mem-001"

    mock_memory.add.assert_called_once()


@patch("app.routers.memory.get_memory")
def test_list_memories(mock_get_memory):
    """GET /memories returns memories from mem0 get_all()."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    # Mock add() for the two create calls
    mock_memory.add.side_effect = [
        {"results": [{"id": "mem-a", "memory": "memory A"}]},
        {"results": [{"id": "mem-b", "memory": "memory B"}]},
    ]

    # Create two memories
    client.post("/api/v1/memories", json={"content": "memory A"})
    client.post("/api/v1/memories", json={"content": "memory B"})

    # Mock get_all() for listing
    mock_memory.get_all.return_value = {
        "results": [
            {"id": "mem-b", "memory": "memory B", "created_at": "2026-05-15T10:00:00", "updated_at": "2026-05-15T10:00:00"},
            {"id": "mem-a", "memory": "memory A", "created_at": "2026-05-15T09:00:00", "updated_at": "2026-05-15T09:00:00"},
        ]
    }

    resp = client.get("/api/v1/memories")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["page"] == 1
    items = data["items"]
    assert len(items) == 2
    assert items[0]["content"] == "memory B"
    assert items[0]["id"] == "mem-b"


@patch("app.routers.memory.get_memory")
def test_update_memory(mock_get_memory):
    """PUT /memories/{id} updates content via mem0."""
    mock_memory = MagicMock()
    mock_memory.add.return_value = {
        "results": [{"id": "mem-xyz", "memory": "original content"}]
    }
    mock_memory.update.return_value = {"id": "mem-xyz", "memory": "updated content"}
    mock_get_memory.return_value = mock_memory

    create_resp = client.post("/api/v1/memories", json={"content": "original content"})
    mem_id = create_resp.json()["id"]

    resp = client.put(f"/api/v1/memories/{mem_id}", json={"content": "updated content"})
    assert resp.status_code == 200
    assert resp.json()["content"] == "updated content"

    mock_memory.update.assert_called_once_with(memory_id=mem_id, data="updated content")


@patch("app.routers.memory.get_memory")
def test_update_memory_not_found(mock_get_memory):
    """PUT /memories/{id} with non-existent id returns 404."""
    mock_memory = MagicMock()
    mock_memory.update.side_effect = Exception("not found")
    mock_get_memory.return_value = mock_memory

    resp = client.put("/api/v1/memories/nonexistent", json={"content": "test"})
    assert resp.status_code == 404


@patch("app.routers.memory.get_memory")
def test_delete_memory(mock_get_memory):
    """DELETE /memories/{id} removes via mem0."""
    mock_memory = MagicMock()
    mock_memory.add.return_value = {
        "results": [{"id": "mem-del", "memory": "to be deleted"}]
    }
    mock_get_memory.return_value = mock_memory

    create_resp = client.post("/api/v1/memories", json={"content": "to be deleted"})
    mem_id = create_resp.json()["id"]

    resp = client.delete(f"/api/v1/memories/{mem_id}")
    assert resp.status_code == 200

    mock_memory.delete.assert_called_once_with(memory_id=mem_id)


@patch("app.routers.memory.get_memory")
def test_delete_memory_not_found(mock_get_memory):
    """DELETE /memories/{id} with non-existent id returns 404."""
    mock_memory = MagicMock()
    mock_memory.delete.side_effect = Exception("not found")
    mock_get_memory.return_value = mock_memory

    resp = client.delete("/api/v1/memories/nonexistent")
    assert resp.status_code == 404


@patch("app.routers.memory.get_memory")
def test_search_memories(mock_get_memory):
    """GET /memories/search?q=xxx returns semantically ranked results from mem0."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    # Mock add() for creating test memories
    mock_memory.add.side_effect = [
        {"results": [{"id": "mem-1", "memory": "我喜欢Python编程"}]},
        {"results": [{"id": "mem-2", "memory": "我的名字叫小明"}]},
    ]

    client.post("/api/v1/memories", json={"content": "我喜欢Python编程"})
    client.post("/api/v1/memories", json={"content": "我的名字叫小明"})

    # Mock search results
    mock_memory.search.return_value = {
        "results": [
            {"id": "mem-2", "memory": "我的名字叫小明", "score": 0.85},
            {"id": "mem-1", "memory": "我喜欢Python编程", "score": 0.65},
        ]
    }

    resp = client.get("/api/v1/memories/search", params={"q": "我叫什么名字"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    items = data["items"]
    assert len(items) == 2
    assert items[0]["content"] == "我的名字叫小明"
    assert items[0]["distance"] == 0.85
    assert items[1]["distance"] == 0.65

    mock_memory.search.assert_called_once()
