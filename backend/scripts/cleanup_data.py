"""
清理脚本：删除 PostgreSQL + Milvus 中的测试对话数据。

用法：
    python scripts/cleanup_data.py              # 交互式确认后清理
    python scripts/cleanup_data.py --force       # 跳过确认直接清理
    python scripts/cleanup_data.py --keep-users  # 保留用户账号
"""
import argparse
import os
import sys

# 确保能找到 backend/app 模块
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# 检查是否在虚拟环境中运行（检测 pymilvus，它不太可能全局安装）
try:
    from pymilvus import MilvusClient  # noqa: F401
except ImportError:
    print("=" * 50)
    print("  错误：未在虚拟环境中运行！")
    print("=" * 50)
    print()
    print("  请使用以下命令：")
    print("    cd backend && venv\\Scripts\\python scripts\\cleanup_data.py")
    print("  或双击：")
    print("    backend\\scripts\\cleanup_data.bat")
    print()
    sys.exit(1)

# 加载 .env
env_path = os.path.join(base_dir, '.env')
if os.path.exists(env_path):
    from dotenv import load_dotenv
    load_dotenv(env_path)

from sqlalchemy import text
from app.db import SessionLocal
from app.services.milvus_store import get_milvus_client, ensure_collection, COLLECTION_NAME


def clean_postgres(keep_users: bool = False) -> dict[str, int]:
    counts = {}
    db = SessionLocal()
    try:
        counts["messages"] = db.execute(text("SELECT count(*) FROM messages")).scalar()
        counts["uploaded_files"] = db.execute(text("SELECT count(*) FROM uploaded_files")).scalar()
        counts["conversations"] = db.execute(text("SELECT count(*) FROM conversations")).scalar()
        counts["user_memories"] = db.execute(text("SELECT count(*) FROM user_memories")).scalar()
        if not keep_users:
            counts["users"] = db.execute(text("SELECT count(*) FROM users")).scalar()

        db.execute(text("DELETE FROM messages"))
        db.execute(text("DELETE FROM uploaded_files"))
        db.execute(text("DELETE FROM conversations"))
        db.execute(text("DELETE FROM user_memories"))
        if not keep_users:
            db.execute(text("DELETE FROM users"))
        db.commit()
    finally:
        db.close()
    return counts


def clean_milvus() -> bool:
    try:
        client = get_milvus_client()
        for coll in [COLLECTION_NAME, "mem0_memories"]:
            if coll in client.list_collections():
                client.drop_collection(coll)
        # 重建 conversation_history
        ensure_collection(client)
        return True
    except Exception as e:
        print(f"  [WARN] Milvus 清理失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="清理 Chat Agent 测试数据")
    parser.add_argument("--force", action="store_true", help="跳过确认直接清理")
    parser.add_argument("--keep-users", action="store_true", help="保留用户账号")
    args = parser.parse_args()

    print("=" * 50)
    print("  Chat Agent — 测试数据清理")
    print("=" * 50)

    # 预览
    db = SessionLocal()
    try:
        msg_count = db.execute(text("SELECT count(*) FROM messages")).scalar()
        file_count = db.execute(text("SELECT count(*) FROM uploaded_files")).scalar()
        conv_count = db.execute(text("SELECT count(*) FROM conversations")).scalar()
        mem_count = db.execute(text("SELECT count(*) FROM user_memories")).scalar()
        user_count = db.execute(text("SELECT count(*) FROM users")).scalar() if not args.keep_users else 0
    finally:
        db.close()

    print(f"\n即将删除：")
    print(f"  PostgreSQL: {conv_count} 个对话, {msg_count} 条消息, {file_count} 个文件"
          f", {mem_count} 条用户记忆"
          f"{f', {user_count} 个用户' if not args.keep_users else ''}")
    print(f"  Milvus: collection 'conversation_history' 和 'mem0_memories' 将清空重建")

    if not args.force:
        print("\n确认删除？此操作不可恢复！")
        try:
            confirm = input("输入 YES 确认: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            sys.exit(1)
        if confirm != "YES":
            print("已取消")
            sys.exit(1)

    # 执行清理
    print("\n正在清理 PostgreSQL...")
    pg_counts = clean_postgres(keep_users=args.keep_users)
    user_info = ""
    if not args.keep_users:
        user_info = f", {pg_counts['users']} 个用户"
    print(f"  已删除: {pg_counts['messages']} 条消息, "
          f"{pg_counts['uploaded_files']} 个文件, "
          f"{pg_counts['conversations']} 个对话, "
          f"{pg_counts['user_memories']} 条用户记忆"
          f"{user_info}")

    print("正在清理 Milvus...")
    if clean_milvus():
        print(f"  已清空并重建 Milvus collections")
    else:
        print("  [WARN] Milvus 清理跳过")

    print("\n清理完成！")


if __name__ == "__main__":
    main()
