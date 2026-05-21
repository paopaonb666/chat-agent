"""Migrate old role=memory records from conversation_history to user_memories + SQL."""
import asyncio
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.embedding import get_dense_embedding
from app.services.memory_store import (
    get_memory_client, ensure_memory_collection, insert_memory,
)
from app.db import SessionLocal
from app.models import UserMemory, User
from app.core.security import get_password_hash

_LOCAL_USER_CACHE: dict[str, int | None] = {"id": None}


def _get_local_user(db) -> User:
    if _LOCAL_USER_CACHE["id"] is not None:
        user = db.query(User).filter(User.id == _LOCAL_USER_CACHE["id"]).first()
        if user:
            return user
    user = db.query(User).order_by(User.id).first()
    if user:
        _LOCAL_USER_CACHE["id"] = user.id
        return user
    user = User(username="local", password_hash=get_password_hash("local"))
    db.add(user)
    db.commit()
    db.refresh(user)
    _LOCAL_USER_CACHE["id"] = user.id
    return user


async def migrate():
    client = get_memory_client()
    collections = client.list_collections()

    if "conversation_history" not in collections:
        print("No conversation_history collection found — nothing to migrate.")
        return

    # Load and query old memory records
    client.load_collection("conversation_history")
    old_records = client.query(
        collection_name="conversation_history",
        filter='role == "memory"',
        output_fields=["content", "user_id"],
        limit=1000,
    )
    print(f"Found {len(old_records)} old memory records in conversation_history")

    if not old_records:
        print("Nothing to migrate.")
        return

    # Ensure target collections
    dim = 1024
    ensure_memory_collection(client, dim=dim)
    client.load_collection("user_memories")

    db = SessionLocal()
    try:
        for i, rec in enumerate(old_records):
            content = rec.get("content", "")
            if not content:
                continue

            # Map old records to the local user (FK constraint: users.id must exist)
            user_id = _get_local_user(db).id

            # Create SQL record
            mem = UserMemory(user_id=user_id, content=content, source="auto_extracted")
            db.add(mem)
            db.commit()
            db.refresh(mem)

            # Get embedding and insert to Milvus
            vec = await get_dense_embedding(content)
            insert_memory(
                client,
                user_id=user_id,
                content=content,
                memory_id=mem.id,
                dense_embedding=vec,
            )
            print(f"  [{i+1}/{len(old_records)}] migrated: {content[:50]}")
    finally:
        db.close()

    print("\nMigration complete!")

    # Verify
    verify = client.query(
        collection_name="user_memories",
        filter="",
        output_fields=["user_id", "memory_id", "content"],
        limit=1000,
    )
    print(f"user_memories now has {len(verify)} records")


if __name__ == "__main__":
    asyncio.run(migrate())
