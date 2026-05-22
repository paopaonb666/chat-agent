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
from app.models import User, UserMemory
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
    """POST /memories writes to PostgreSQL first, then syncs to Milvus."""
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
    # ID is now a string of the PostgreSQL int id
    assert data["id"].isdigit()

    # Verify PostgreSQL was written
    db = TestingSessionLocal()
    try:
        pg_mem = db.query(UserMemory).filter(UserMemory.id == int(data["id"])).first()
        assert pg_mem is not None
        assert pg_mem.content == "我喜欢Python编程"
        assert pg_mem.source == "manual"
    finally:
        db.close()

    # Verify Milvus sync was called
    mock_memory.add.assert_called_once()


@patch("app.routers.memory.get_memory")
def test_list_memories(mock_get_memory):
    """GET /memories returns memories from PostgreSQL with DB-level pagination."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    # Write directly to PostgreSQL (simulating existing data)
    db = TestingSessionLocal()
    try:
        user = User(username="testuser", password_hash="hash", role="user")
        db.add(user)
        db.flush()
        db.add(UserMemory(user_id=user.id, content="memory A", source="manual"))
        db.add(UserMemory(user_id=user.id, content="memory B", source="auto_extracted"))
        db.commit()
    finally:
        db.close()

    resp = client.get("/api/v1/memories")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["page"] == 1
    items = data["items"]
    assert len(items) == 2
    contents = {m["content"] for m in items}
    assert contents == {"memory A", "memory B"}
    for m in items:
        assert m["id"].isdigit()
        assert m["source"] in ("manual", "auto_extracted")


@patch("app.routers.memory.get_memory")
def test_list_memories_pagination(mock_get_memory):
    """GET /memories supports DB-level pagination."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    db = TestingSessionLocal()
    try:
        user = User(username="testuser", password_hash="hash", role="user")
        db.add(user)
        db.flush()
        for i in range(5):
            db.add(UserMemory(user_id=user.id, content=f"memory {i}", source="manual"))
        db.commit()
    finally:
        db.close()

    resp = client.get("/api/v1/memories?page=1&page_size=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert data["total_pages"] == 3
    assert len(data["items"]) == 2

    resp2 = client.get("/api/v1/memories?page=3&page_size=2")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["items"]) == 1


@patch("app.routers.memory.get_memory")
def test_update_memory(mock_get_memory):
    """PUT /memories/{id} updates PostgreSQL and syncs to Milvus."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    # Create via API (writes PG + Milvus)
    mock_memory.add.return_value = {
        "results": [{"id": "mem-xyz", "memory": "original content"}]
    }

    create_resp = client.post("/api/v1/memories", json={"content": "original content"})
    mem_id = create_resp.json()["id"]
    assert mem_id.isdigit()

    # Update
    resp = client.put(f"/api/v1/memories/{mem_id}", json={"content": "updated content"})
    assert resp.status_code == 200
    assert resp.json()["content"] == "updated content"
    assert resp.json()["id"] == mem_id

    # Verify PostgreSQL was updated
    db = TestingSessionLocal()
    try:
        pg_mem = db.query(UserMemory).filter(UserMemory.id == int(mem_id)).first()
        assert pg_mem is not None
        assert pg_mem.content == "updated content"
    finally:
        db.close()


@patch("app.routers.memory.get_memory")
def test_update_memory_not_found(mock_get_memory):
    """PUT /memories/{id} with non-existent int id returns 404."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    resp = client.put("/api/v1/memories/99999", json={"content": "test"})
    assert resp.status_code == 404


@patch("app.routers.memory.get_memory")
def test_delete_memory(mock_get_memory):
    """DELETE /memories/{id} removes from PostgreSQL and syncs to Milvus."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    # Create via API
    mock_memory.add.return_value = {
        "results": [{"id": "mem-del", "memory": "to be deleted"}]
    }

    create_resp = client.post("/api/v1/memories", json={"content": "to be deleted"})
    mem_id = create_resp.json()["id"]
    assert mem_id.isdigit()

    resp = client.delete(f"/api/v1/memories/{mem_id}")
    assert resp.status_code == 200
    assert resp.json() == {"detail": "Memory deleted"}

    # Verify PostgreSQL record is gone
    db = TestingSessionLocal()
    try:
        pg_mem = db.query(UserMemory).filter(UserMemory.id == int(mem_id)).first()
        assert pg_mem is None
    finally:
        db.close()


@patch("app.routers.memory.get_memory")
def test_delete_memory_not_found(mock_get_memory):
    """DELETE /memories/{id} with non-existent int id returns 404."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    resp = client.delete("/api/v1/memories/99999")
    assert resp.status_code == 404


@patch("app.routers.memory.get_memory")
def test_search_memories(mock_get_memory):
    """GET /memories/search?q=xxx returns semantically ranked results from Milvus."""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

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
