import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.memory_client import get_memory
from app.db import SessionLocal
from app.models import User

SAMPLE_MEMORIES = [
    "用户喜欢使用 Python 进行后端开发，熟悉 FastAPI 和 SQLAlchemy。",
    "用户对 AI 和 LLM 应用开发有浓厚兴趣，正在学习 RAG 和 Agent 架构。",
    "用户的工作环境是 Windows 11 + Git Bash，习惯使用 Unix 风格的命令行工具。",
    "用户偏好简洁的代码风格，不喜欢过度工程化和无意义的注释。",
    "用户同时在做学习和真实工程开发两件事，学习项目在 E:\\ai_study\\ 目录下。",
    "用户喜欢通过实际项目来学习新技术，而不是只看文档和教程。",
    "用户对数据库优化和性能调优有经验，关注查询效率和索引设计。",
]


def run(user_id: str = "1", count: int | None = None, all_users: bool = False) -> int:
    """向 mem0_memories 写入示例记忆数据。

    Args:
        user_id: 目标用户 ID（字符串）。
        count: 写入条数，None 表示全部。
        all_users: 是否给数据库中所有用户各写 count 条。

    Returns:
        实际写入的条数。
    """
    memory = get_memory()
    if memory is None:
        print("ERROR: mem0 初始化失败，请检查 Ollama 和 Milvus 是否运行。")
        return 0

    if all_users:
        db = SessionLocal()
        try:
            users = db.query(User).all()
            user_ids = [str(u.id) for u in users]
        finally:
            db.close()
        if not user_ids:
            print("WARNING: 数据库中没有用户。")
            return 0
        print(f"发现 {len(user_ids)} 个用户: {', '.join(user_ids)}")
    else:
        user_ids = [user_id]

    per_user = count if count is not None else 2
    total_written = 0

    for idx, uid in enumerate(user_ids):
        written = 0
        for i in range(per_user):
            content = SAMPLE_MEMORIES[(idx * per_user + i) % len(SAMPLE_MEMORIES)]
            try:
                memory.add(
                    [{"role": "user", "content": content}],
                    user_id=uid,
                    metadata={"source": "manual"},
                )
                print(f"  [OK] user_id={uid}: {content[:50]}...")
                written += 1
            except Exception as e:
                print(f"  [FAIL] user_id={uid}: {content[:50]}... -> {e}")
        total_written += written
        print(f"  user_id={uid} 写入 {written}/{per_user} 条")

    print(f"\n总计写入: {total_written} 条")
    return total_written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed sample memories into mem0")
    parser.add_argument("--user-id", default="1", help="目标用户 ID (默认: 1)")
    parser.add_argument("--count", type=int, default=None, help="每个用户写入条数 (默认: 2)")
    parser.add_argument("--all-users", action="store_true", help="给数据库中所有用户写入")
    args = parser.parse_args()

    run(user_id=args.user_id, count=args.count, all_users=args.all_users)
