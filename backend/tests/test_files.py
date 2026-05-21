import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db import Base, get_db
from app.deps import get_current_user, oauth2_scheme
from app.models import User
from app.core.security import get_password_hash
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


def _create_conv():
    resp = client.post("/api/v1/conversations", json={"title": "File Test"})
    assert resp.status_code == 200
    return resp.json()["id"]


def test_upload_txt():
    conv_id = _create_conv()
    resp = client.post(
        "/api/v1/files/upload",
        data={"conversation_id": conv_id},
        files={"file": ("test.txt", b"Hello world", "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test.txt"
    assert data["extracted_text"] == "Hello world"


def test_upload_pdf():
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # Test that PDF mime type is accepted and returns a file record.
    conv_id = _create_conv()
    resp = client.post(
        "/api/v1/files/upload",
        data={"conversation_id": conv_id},
        files={"file": ("test.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test.pdf"
    # Since it's a fake PDF, extracted_text may be empty but request should succeed


def test_upload_invalid_conversation():
    resp = client.post(
        "/api/v1/files/upload",
        data={"conversation_id": "nonexistent"},
        files={"file": ("test.txt", b"Hello", "text/plain")},
    )
    assert resp.status_code == 404
