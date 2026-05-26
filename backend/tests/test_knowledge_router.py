import pytest
from unittest.mock import patch, MagicMock
from app.models import KnowledgeDocument, User
from app.core.security import get_password_hash


def test_upload_document_admin(client, db_session):
    async def mock_enqueue_job(*args, **kwargs):
        return None
    mock_pool = MagicMock()
    mock_pool.enqueue_job = mock_enqueue_job
    with patch("app.routers.knowledge.get_arq_pool", return_value=mock_pool):
        response = client.post(
            "/api/v1/knowledge/documents/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["status"] == "pending"


def test_upload_document_forbidden_for_non_admin(client, db_session):
    user = User(username="normaluser", password_hash=get_password_hash("pass"), role="user")
    db_session.add(user)
    db_session.commit()

    def override_get_current_user_normal():
        return user

    from app.deps import get_current_user
    from app.main import app
    original_override = app.dependency_overrides.get(get_current_user)
    app.dependency_overrides[get_current_user] = override_get_current_user_normal
    try:
        response = client.post(
            "/api/v1/knowledge/documents/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 403
    finally:
        if original_override:
            app.dependency_overrides[get_current_user] = original_override
        else:
            del app.dependency_overrides[get_current_user]


def test_list_documents(client, db_session):
    user = db_session.query(User).filter(User.username == "testuser").first()
    if not user:
        user = User(username="testuser", password_hash=get_password_hash("testpass"), role="admin")
        db_session.add(user)
        db_session.commit()
    doc1 = KnowledgeDocument(
        filename="public.txt",
        mime_type="text/plain",
        file_size=5,
        owner_id=user.id,
        visibility="public",
        status="completed",
        path="",
    )
    doc2 = KnowledgeDocument(
        filename="private.txt",
        mime_type="text/plain",
        file_size=5,
        owner_id=user.id,
        visibility="private",
        status="completed",
        path="",
    )
    db_session.add_all([doc1, doc2])
    db_session.commit()

    response = client.get("/api/v1/knowledge/documents")
    assert response.status_code == 200
    data = response.json()
    # testuser is admin, sees all documents
    assert data["total"] == 2


def test_delete_document(client, db_session):
    user = db_session.query(User).filter(User.username == "testuser").first()
    if not user:
        user = User(username="testuser", password_hash=get_password_hash("testpass"), role="admin")
        db_session.add(user)
        db_session.commit()
    doc = KnowledgeDocument(
        filename="del.txt",
        mime_type="text/plain",
        file_size=5,
        owner_id=user.id,
        visibility="public",
        status="completed",
        path="",
    )
    db_session.add(doc)
    db_session.commit()

    with patch("app.routers.knowledge.delete_by_document"):
        response = client.delete(f"/api/v1/knowledge/documents/{doc.id}")
        assert response.status_code == 200


def test_search_knowledge(client, db_session):
    user = db_session.query(User).filter(User.username == "testuser").first()
    if not user:
        user = User(username="testuser", password_hash=get_password_hash("testpass"), role="admin")
        db_session.add(user)
        db_session.commit()
    doc = KnowledgeDocument(
        filename="searchable.txt",
        mime_type="text/plain",
        file_size=5,
        owner_id=user.id,
        visibility="public",
        status="completed",
        vector_indexed=True,
        path="",
    )
    db_session.add(doc)
    db_session.commit()

    with patch("app.routers.knowledge.get_milvus_client") as mock_milvus, \
         patch("app.services.embedding.get_dense_embedding") as mock_emb, \
         patch("app.services.knowledge_milvus.search_knowledge_base") as mock_search:
        mock_emb.return_value = [0.1] * 1024
        mock_search.return_value = [
            {"id": "chunk1", "content": "relevant info", "distance": 0.9, "document_id": doc.id}
        ]
        response = client.post("/api/v1/knowledge/search", json={"query": "test", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["content"] == "relevant info"
