import os
import sys
import random
import asyncio
import numpy as np

os.chdir('E:/ai_study/chat-agent/backend')
sys.path.insert(0, 'E:/ai_study/chat-agent/backend')

from dotenv import load_dotenv
load_dotenv(dotenv_path='E:/ai_study/chat-agent/backend/.env')

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.db import engine, SessionLocal
from app import models
from datetime import datetime, timezone
import uuid

from pymilvus import MilvusClient

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DIM = 1024

# ---------- helpers ----------
def random_unit_vector(dim: int = DIM) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()

# ---------- clear data ----------
def clear_postgres():
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM messages"))
        conn.execute(text("DELETE FROM uploaded_files"))
        conn.execute(text("DELETE FROM conversations"))
        conn.execute(text("DELETE FROM user_memories"))
    print("[PG] Cleared messages, uploaded_files, conversations, user_memories")

def clear_milvus():
    client = MilvusClient(uri=MILVUS_URI)
    for coll in ["conversation_history", "mem0_memories", "user_memories", "mem0migrations"]:
        if coll in client.list_collections():
            client.drop_collection(coll)
            print(f"[Milvus] Dropped collection: {coll}")
    client.close()

# ---------- seed ----------
USER_SAMPLES = [
    {"id": 2, "username": "local"},
    {"id": 3, "username": "test"},
    {"id": 4, "username": "admin"},
    {"id": 5, "username": "testlogin"},
]

CONV_TITLES = [
    "如何学习Python", "Docker入门", "数据库优化", "前端框架对比", "微服务架构",
    "Redis缓存策略", "Linux常用命令", "CI/CD实践", "代码审查规范", "性能调优技巧",
]

USER_MSGS = [
    "能详细讲讲吗？", "有没有什么推荐的学习资源？", "实际项目中应该怎么用？",
    "和其他方案相比优缺点是什么？", "初学者容易犯哪些错误？",
]

ASSISTANT_MSGS = [
    "当然，这个话题可以从几个维度来看...",
    "推荐你先掌握基础概念，然后动手做一个小项目。",
    "在实际项目中，建议遵循最佳实践，比如保持代码简洁、做好错误处理。",
    "相比其他方案，它的优势在于生态成熟、社区活跃。",
    "初学者常见的错误包括过度设计、忽略文档、以及不重视测试。",
]

MEMORY_CONTENTS = [
    "用户喜欢通过实际项目来学习新技术",
    "用户关注代码质量和工程最佳实践",
    "用户偏好中文技术资料",
    "用户对性能优化和架构设计感兴趣",
    "用户习惯在Linux环境下开发",
    "用户重视自动化测试和CI/CD流程",
]

def seed_data():
    db = SessionLocal()
    client = MilvusClient(uri=MILVUS_URI)

    # Ensure conversation_history collection exists
    from app.services.milvus_store import ensure_collection
    ensure_collection(client, dim=DIM)

    for user in USER_SAMPLES:
        user_id = user["id"]
        print(f"\n[Seed] User {user['username']} (id={user_id})")

        # Insert 10 conversations
        for i in range(10):
            conv_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            conv = models.Conversation(
                id=conv_id,
                user_id=user_id,
                title=CONV_TITLES[i],
                model="deepseek-chat",
                created_at=now,
                updated_at=now,
            )
            db.add(conv)
            db.flush()  # get conv.id without commit

            # Each conversation: 1 user msg + 1 assistant msg
            msg_user = models.Message(
                conversation_id=conv_id,
                role="user",
                content=random.choice(USER_MSGS),
                created_at=now,
            )
            msg_assistant = models.Message(
                conversation_id=conv_id,
                role="assistant",
                content=random.choice(ASSISTANT_MSGS),
                created_at=now,
            )
            db.add(msg_user)
            db.add(msg_assistant)
            db.flush()

            # Insert into Milvus for both messages
            import time as _time
            ts = int(_time.time())
            for msg in [msg_user, msg_assistant]:
                client.insert(
                    collection_name="conversation_history",
                    data=[{
                        "conversation_id": conv_id,
                        "user_id": user_id,
                        "role": msg.role,
                        "content": msg.content,
                        "dense_embedding": random_unit_vector(),
                        "timestamp": ts,
                        "message_id": msg.id,
                    }]
                )

        # Insert 3 user memories
        selected_memories = random.sample(MEMORY_CONTENTS, 3)
        for mem_content in selected_memories:
            mem = models.UserMemory(
                user_id=user_id,
                content=mem_content,
                source="manual",
            )
            db.add(mem)

    db.commit()
    db.close()
    client.close()
    print("\n[Done] Seeded data for all users")

if __name__ == "__main__":
    print("=== Step 1: Clear existing data ===")
    clear_postgres()
    clear_milvus()
    print("\n=== Step 2: Seed new data ===")
    seed_data()
    print("\n=== All done ===")
