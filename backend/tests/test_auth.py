import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db import Base, get_db
from app.core.security import verify_password, get_password_hash, create_access_token, decode_access_token
from app import models  # noqa: F401 — registers tables with Base.metadata

# Setup test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool)
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


def test_password_hash():
    plain = "secret123"
    hashed = get_password_hash(plain)
    assert verify_password(plain, hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_token():
    token = create_access_token({"sub": "alice"})
    payload = decode_access_token(token)
    assert payload["sub"] == "alice"


def test_register():
    resp = client.post("/api/v1/auth/register", json={
        "username": "alice",
        "password": "secret123",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == "alice"
    assert "id" in data
    assert "password_hash" not in data


def test_register_duplicate():
    client.post("/api/v1/auth/register", json={"username": "bob", "password": "secret123"})
    resp = client.post("/api/v1/auth/register", json={"username": "bob", "password": "secret123"})
    assert resp.status_code == 400


def test_login():
    client.post("/api/v1/auth/register", json={"username": "charlie", "password": "secret123"})
    resp = client.post("/api/v1/auth/login", data={
        "username": "charlie",
        "password": "secret123",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password():
    client.post("/api/v1/auth/register", json={"username": "dave", "password": "secret123"})
    resp = client.post("/api/v1/auth/login", data={
        "username": "dave",
        "password": "wrong",
    })
    assert resp.status_code == 401
