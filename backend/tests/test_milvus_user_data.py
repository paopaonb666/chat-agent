import pytest
from pymilvus import MilvusClient
from app.core.config import settings


@pytest.fixture
def milvus_client():
    client = MilvusClient(uri=settings.milvus_uri)
    yield client


def test_milvus_connection(milvus_client):
    """验证能连上 Milvus"""
    collections = milvus_client.list_collections()
    assert isinstance(collections, list)


def test_mem0_collection_exists(milvus_client):
    """验证 mem0_memories collection 存在"""
    collections = milvus_client.list_collections()
    assert "mem0_memories" in collections, (
        f"mem0_memories collection 不存在。现有 collections: {collections}"
    )


def test_mem0_collection_schema(milvus_client):
    """打印 mem0_memories 的 schema 和字段信息"""
    schema = milvus_client.describe_collection("mem0_memories")
    fields = [f.get("name") for f in schema.get("fields", [])]
    print(f"\n[SCHEMA] mem0_memories fields: {fields}")
    for f in schema.get("fields", []):
        print(f"  - {f.get('name')}: type={f.get('type')}, params={f.get('params', {})}")
    assert "id" in fields


def test_mem0_total_count(milvus_client):
    """统计 mem0_memories 总数据条数"""
    if "mem0_memories" not in milvus_client.list_collections():
        pytest.skip("mem0_memories 不存在")

    # 只查询标量字段，排除 vectors/sparse 等向量字段（Milvus 不允许 query 取出原始向量）
    scalar_fields = ["id", "metadata", "text"]

    results = milvus_client.query(
        collection_name="mem0_memories",
        filter="",
        output_fields=scalar_fields,
        limit=10000,
    )
    total = len(results)
    print(f"\n[COUNT] mem0_memories 总数据条数: {total}")
    assert total >= 0  # 只记录，不强制要求有数据


def test_mem0_data_by_user(milvus_client):
    """按 user_id 统计 mem0_memories 中的数据分布"""
    if "mem0_memories" not in milvus_client.list_collections():
        pytest.skip("mem0_memories 不存在")

    scalar_fields = ["id", "metadata", "text"]

    results = milvus_client.query(
        collection_name="mem0_memories",
        filter="",
        output_fields=scalar_fields,
        limit=10000,
    )

    if not results:
        print("\n[DATA] mem0_memories 中没有任何数据")
        return

    # 按 user_id 分组统计（mem0 把 user_id 存在 metadata JSON 里）
    user_counts = {}
    for r in results:
        metadata = r.get("metadata", {}) or {}
        uid = metadata.get("user_id", "N/A") if isinstance(metadata, dict) else "N/A"
        user_counts[uid] = user_counts.get(uid, 0) + 1

    print(f"\n[USER STATS] 按 user_id 分布:")
    for uid, count in sorted(user_counts.items(), key=lambda x: -x[1]):
        print(f"  user_id={uid}: {count} 条")

    # 打印前 3 条样本
    print(f"\n[SAMPLES] 前 3 条数据:")
    for r in results[:3]:
        # mem0 用 text 字段存记忆内容，user_id 在 metadata JSON 里
        metadata = r.get("metadata", {}) or {}
        uid = metadata.get("user_id", "N/A") if isinstance(metadata, dict) else "N/A"
        print(f"  id={r.get('id')}, user_id={uid}, "
              f"text={str(r.get('text', ''))[:100]}...")


def test_conversation_history_collection_exists(milvus_client):
    """同时检查 conversation_history collection（RAG 用的）"""
    collections = milvus_client.list_collections()
    assert "conversation_history" in collections, (
        f"conversation_history 不存在。现有 collections: {collections}"
    )


def test_conversation_history_data_by_user(milvus_client):
    """按 user_id 统计 conversation_history 中的数据"""
    if "conversation_history" not in milvus_client.list_collections():
        pytest.skip("conversation_history 不存在")

    results = milvus_client.query(
        collection_name="conversation_history",
        filter="",
        output_fields=["user_id", "conversation_id", "role", "content"],
        limit=10000,
    )

    user_counts = {}
    for r in results:
        uid = r.get("user_id", -1)
        user_counts[uid] = user_counts.get(uid, 0) + 1

    print(f"\n[CONV HISTORY] conversation_history 总条数: {len(results)}")
    print(f"[CONV HISTORY] 按 user_id 分布:")
    for uid, count in sorted(user_counts.items(), key=lambda x: -x[1]):
        print(f"  user_id={uid}: {count} 条")
