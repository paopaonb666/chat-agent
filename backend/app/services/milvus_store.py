import os
from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "conversation_history"
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")


def get_milvus_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)


def ensure_collection(client: MilvusClient, dim: int = 768) -> None:
    if COLLECTION_NAME in client.list_collections():
        return
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("conversation_id", DataType.VARCHAR, max_length=36)
    schema.add_field("user_id", DataType.INT64)
    schema.add_field("role", DataType.VARCHAR, max_length=10)
    schema.add_field("content", DataType.VARCHAR, max_length=65535)
    schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("timestamp", DataType.INT64)
    schema.add_field("message_id", DataType.INT64)
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="dense_embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=COLLECTION_NAME, index_params=idx)


def insert_message(
    client: MilvusClient,
    *,
    conversation_id: str,
    user_id: int | None,
    role: str,
    content: str,
    message_id: int,
    dense_embedding: list[float],
    timestamp: int | None = None,
) -> None:
    import time

    client.insert(
        collection_name=COLLECTION_NAME,
        data=[
            {
                "conversation_id": conversation_id,
                "user_id": user_id if user_id is not None else -1,
                "role": role,
                "content": content,
                "dense_embedding": dense_embedding,
                "timestamp": timestamp or int(time.time()),
                "message_id": message_id,
            }
        ],
    )


def search_dense(
    client: MilvusClient,
    dense_embedding: list[float],
    user_id: int | None,
    top_k: int = 10,
) -> list[dict]:
    expr = None if user_id is None else f"user_id == {user_id}"
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[dense_embedding],
        anns_field="dense_embedding",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["conversation_id", "role", "content", "message_id", "timestamp"],
        filter=expr,
    )
    hits = []
    for group in res:
        for hit in group:
            hits.append(
                {
                    "id": hit["id"],
                    "distance": hit["distance"],
                    "conversation_id": hit["entity"].get("conversation_id"),
                    "role": hit["entity"].get("role"),
                    "content": hit["entity"].get("content"),
                    "message_id": hit["entity"].get("message_id"),
                    "timestamp": hit["entity"].get("timestamp"),
                }
            )
    return hits
