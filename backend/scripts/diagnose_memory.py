"""
诊断脚本：检查跨会话记忆的数据链路是否通畅。

用法：
    cd backend && venv\Scripts\python scripts\diagnose_memory.py
"""
import os
import sys

# Fix Windows console encoding for UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

env_path = os.path.join(base_dir, '.env')
if os.path.exists(env_path):
    from dotenv import load_dotenv
    load_dotenv(env_path)

from app.db import SessionLocal
from app.services.milvus_store import get_milvus_client, COLLECTION_NAME
from app.services.embedding import get_dense_embedding
from app.services.hybrid_search import hybrid_search


async def main():
    print("=" * 60)
    print("  跨会话记忆诊断")
    print("=" * 60)

    # 1. PostgreSQL 数据量
    print("\n[1] PostgreSQL 数据量：")
    db = SessionLocal()
    try:
        from sqlalchemy import text
        conv_count = db.execute(text("SELECT count(*) FROM conversations")).scalar()
        msg_count = db.execute(text("SELECT count(*) FROM messages")).scalar()
        print(f"    对话数: {conv_count}")
        print(f"    消息数: {msg_count}")

        if msg_count > 0:
            print("\n    最近 5 条消息：")
            rows = db.execute(text(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT 5"
            )).fetchall()
            for r in rows:
                content = r[1][:60] + "..." if len(r[1]) > 60 else r[1]
                print(f"      [{r[0]}] {content}")
    finally:
        db.close()

    # 2. Milvus collection 状态
    print(f"\n[2] Milvus collection '{COLLECTION_NAME}'：")
    try:
        client = get_milvus_client()
        collections = client.list_collections()
        if COLLECTION_NAME not in collections:
            print(f"    [ERROR] Collection '{COLLECTION_NAME}' 不存在！")
            return
        print(f"    状态: 存在")

        stats = client.get_collection_stats(COLLECTION_NAME)
        row_count = stats.get("row_count", 0)
        print(f"    向量数: {row_count}")

        if row_count == 0:
            print(f"    [WARN] Milvus 中没有任何向量数据！")

        # 3. 测试 Ollama Embedding API
        print("\n[3] Ollama Embedding API：")
        try:
            test_vec = await get_dense_embedding("测试文本")
            print(f"    状态: OK")
            print(f"    向量维度: {len(test_vec)}")
        except Exception as e:
            print(f"    [ERROR] Embedding 失败: {e}")
            print(f"    请检查：1) Ollama 是否启动  2) qwen3-embedding:0.6b 是否已 pull")
            return

        # 4. 如果 Milvus 为空但 PG 有数据，说明索引全部失败了
        if row_count == 0 and msg_count > 0:
            print("\n[4] 尝试手动索引一条消息到 Milvus：")
            try:
                db = SessionLocal()
                try:
                    from sqlalchemy import text
                    row = db.execute(
                        text("SELECT id, conversation_id, role, content FROM messages ORDER BY id DESC LIMIT 1")
                    ).fetchone()
                    if row:
                        msg_id, conv_id, role, content = row
                        vec = await get_dense_embedding(content)
                        client = get_milvus_client()
                        from app.services.milvus_store import insert_message
                        insert_message(
                            client,
                            conversation_id=str(conv_id),
                            user_id=-1,
                            role=role,
                            content=content,
                            message_id=msg_id,
                            dense_embedding=vec,
                        )
                        print(f"    手动索引成功: [{role}] {content[:40]}...")
                        # 重新统计
                        stats = client.get_collection_stats(COLLECTION_NAME)
                        print(f"    当前向量数: {stats.get('row_count', 0)}")
                finally:
                    db.close()
            except Exception as e:
                print(f"    [ERROR] 手动索引失败: {e}")
                import traceback
                traceback.print_exc()
            return

        # 5. 测试检索
        print("\n[5] 模拟检索 '我是谁'：")
        query = "我是谁"
        vec = await get_dense_embedding(query)
        print(f"    查询: '{query}'")
        print(f"    向量维度: {len(vec)}")

        results = await hybrid_search(query, vec, user_id=None, top_k=5)
        print(f"    检索命中: {len(results)} 条")

        if results:
            print("\n    检索结果（按相关性排序）：")
            for i, r in enumerate(results, 1):
                role = r.get("role", "?")
                content = r.get("content", "")[:80]
                print(f"      [{i}] [{role}] {content}")
        else:
            print("    [WARN] 未检索到任何相关内容！")

    except Exception as e:
        print(f"    [ERROR] {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("  诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
