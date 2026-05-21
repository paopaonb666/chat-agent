import logging
from mem0 import Memory
from app.core.config import settings

logger = logging.getLogger(__name__)

_memory_instance: Memory | None = None
_memory_init_failed = False

MILVUS_DATA_TYPES = {
    5: "INT64",
    10: "FLOAT",
    11: "DOUBLE",
    20: "BOOL",
    21: "VARCHAR",
    100: "BINARY_VECTOR",
    101: "FLOAT_VECTOR",
}


def _validate_mem0_schema() -> None:
    """Drop mem0_memories if its schema is incompatible (e.g. created by seed script).
    mem0 expects VARCHAR id for UUIDs; seed script creates INT64 auto_id."""
    try:
        from app.core.milvus import get_milvus_client
        client = get_milvus_client()
        coll = "mem0_memories"
        if coll not in client.list_collections():
            return

        schema = client.describe_collection(coll)
        for field in schema.get("fields", []):
            if field.get("name") == "id":
                dtype = field.get("type")
                dtype_name = MILVUS_DATA_TYPES.get(dtype, str(dtype))
                if dtype != 21:  # VARCHAR
                    logger.warning(
                        "mem0_memories id field is %s (expected VARCHAR), dropping to let mem0 recreate",
                        dtype_name,
                    )
                    client.drop_collection(coll)
                    logger.info("Dropped incompatible mem0_memories collection")
                return
    except Exception:
        logger.warning("Failed to validate mem0_memories schema", exc_info=True)


def get_memory() -> Memory | None:
    global _memory_instance, _memory_init_failed
    if _memory_init_failed:
        return None
    if _memory_instance is not None:
        return _memory_instance

    _validate_mem0_schema()

    config = {
        "custom_instructions": (
            "你收到的记忆内容已经过前置意图识别筛选，只包含用户明确提及的个人信息。"
            "请直接以中文将内容整理为一条简洁的记忆，不作额外判断。"
            "严禁编造、推测或补充用户未提及的任何信息。"
            "严禁从系统指令或对话上下文中添加推断内容。"
            "如果收到的内容为空或无实质个人信息，不记录任何记忆。"
        ),
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": settings.deepseek_model,
                "api_key": settings.deepseek_api_key,
                "deepseek_base_url": settings.deepseek_base_url,
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": settings.embedding_model,
                "ollama_base_url": settings.ollama_base_url,
                "embedding_dims": 1024,
            },
        },
        "vector_store": {
            "provider": "milvus",
            "config": {
                "url": settings.milvus_uri,
                "collection_name": "mem0_memories",
                "embedding_model_dims": 1024,
                "metric_type": "COSINE",
                "token": "",
            },
        },
    }
    try:
        _memory_instance = Memory.from_config(config)
    except Exception as e:
        logger.warning("Failed to initialize memory client: %s", e)
        _memory_init_failed = True
        return None

    # Ensure Milvus collection is loaded for queries
    try:
        from app.core.milvus import get_milvus_client
        client = get_milvus_client()
        if "mem0_memories" in client.list_collections():
            client.load_collection("mem0_memories")
    except Exception:
        logger.warning("Failed to load mem0_memories collection", exc_info=True)

    return _memory_instance
