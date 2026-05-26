import logging
from pymilvus import MilvusClient, DataType

logger = logging.getLogger(__name__)
KB_COLLECTION = "knowledge_base"


def ensure_kb_collection(client: MilvusClient, dim: int = 1024) -> None:
    if KB_COLLECTION in client.list_collections():
        schema = client.describe_collection(KB_COLLECTION)
        for field in schema.get("fields", []):
            if field.get("name") == "embedding":
                existing_dim = field.get("params", {}).get("dim")
                if existing_dim == dim:
                    return
                client.drop_collection(KB_COLLECTION)
                break
        else:
            return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, max_length=36, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("document_id", DataType.VARCHAR, max_length=36)
    schema.add_field("content", DataType.VARCHAR, max_length=65535)
    schema.add_field("meta_json", DataType.VARCHAR, max_length=65535)
    schema.add_field("user_id", DataType.INT64)
    client.create_collection(collection_name=KB_COLLECTION, schema=schema)
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=KB_COLLECTION, index_params=idx)


def insert_knowledge_chunks(client: MilvusClient, chunks: list[dict]) -> None:
    client.insert(collection_name=KB_COLLECTION, data=chunks)


def search_knowledge_base(
    client: MilvusClient,
    dense_embedding: list[float],
    document_ids: list[str] | None,
    top_k: int = 5,
) -> list[dict]:
    kwargs: dict = {
        "collection_name": KB_COLLECTION,
        "data": [dense_embedding],
        "anns_field": "embedding",
        "search_params": {"metric_type": "COSINE", "params": {"ef": 64}},
        "limit": top_k,
        "output_fields": ["document_id", "content", "meta_json"],
    }
    if document_ids:
        ids_expr = ", ".join(f'"{d}"' for d in document_ids)
        kwargs["filter"] = f'document_id in [{ids_expr}]'
    res = client.search(**kwargs)
    hits = []
    for group in res:
        for hit in group:
            hits.append(
                {
                    "id": hit["id"],
                    "distance": hit["distance"],
                    "document_id": hit["entity"].get("document_id"),
                    "content": hit["entity"].get("content"),
                    "meta_json": hit["entity"].get("meta_json"),
                }
            )
    return hits


def delete_by_document(client: MilvusClient, document_id: str) -> None:
    client.delete(collection_name=KB_COLLECTION, filter=f'document_id == "{document_id}"')


def delete_by_documents(client: MilvusClient, document_ids: list[str]) -> None:
    for doc_id in document_ids:
        delete_by_document(client, doc_id)
