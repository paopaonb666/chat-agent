import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, User, Conversation, Message

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_create_user(db_session):
    user = User(username="alice", password_hash="hashed_password")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    assert user.id is not None
    assert user.username == "alice"


def test_user_conversation_relationship(db_session):
    user = User(username="bob", password_hash="hashed_password")
    db_session.add(user)
    db_session.commit()

    conv = Conversation(title="Test Chat", model="deepseek-chat", owner=user)
    db_session.add(conv)
    db_session.commit()

    assert conv.id is not None
    assert conv.user_id == user.id
    assert conv.owner.username == "bob"


def test_conversation_message_relationship(db_session):
    user = User(username="charlie", password_hash="hashed_password")
    db_session.add(user)
    db_session.commit()

    conv = Conversation(title="Msg Test", model="deepseek-chat", owner=user)
    db_session.add(conv)
    db_session.commit()

    msg = Message(role="user", content="hello", conversation=conv)
    db_session.add(msg)
    db_session.commit()

    assert msg.id is not None
    assert msg.conversation_id == conv.id
    assert msg.conversation.title == "Msg Test"


def test_user_unique_username(db_session):
    user1 = User(username="dave", password_hash="hash1")
    db_session.add(user1)
    db_session.commit()

    user2 = User(username="dave", password_hash="hash2")
    db_session.add(user2)
    with pytest.raises(Exception):
        db_session.commit()
