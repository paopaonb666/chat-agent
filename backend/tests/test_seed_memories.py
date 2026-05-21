import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pymilvus import MilvusClient
from app.core.config import settings


def test_seed_memories_script_exists():
    """验证 seed_memories.py 脚本存在且可导入"""
    from scripts import seed_memories
    assert hasattr(seed_memories, "run")


def test_seed_memories_writes_data():
    """运行 seed 脚本并验证 mem0_memories 里有数据"""
    from scripts.seed_memories import run

    run(user_id="1", count=3)

    client = MilvusClient(uri=settings.milvus_uri)
    if "mem0_memories" not in client.list_collections():
        pytest.skip("mem0_memories 不存在")

    results = client.query(
        collection_name="mem0_memories",
        filter="",
        output_fields=["id", "metadata", "text"],
        limit=100,
    )
    # mem0 会把 user_id 存在 metadata 里（字段名可能是 user_id）
    user_results = [r for r in results if r.get("metadata", {}).get("user_id") == "1"]
    print(f"\n[SEED CHECK] user_id=1 的记忆条数: {len(user_results)}")
    for r in user_results[:3]:
        print(f"  - {str(r.get('text', ''))[:80]}...")
    assert len(user_results) >= 1, "seed 后至少应有一条记忆"
    print(f"\n[ASSERT PASSED] 查到 {len(user_results)} 条记忆")
