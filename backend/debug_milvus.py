"""Debug script: query Milvus with "我是谁" and show what comes back."""
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
    get_memory_client,
    ensure_memory_collection,
    search_memories,
    list_all_memories,
)


async def main():
    query_text = "我是谁"
    client = get_memory_client()
    collections = client.list_collections()
    print(f"Milvus collections: {collections}")

    if "user_memories" not in collections:
        print("user_memories collection does not exist — nothing to search.")
        return

    ensure_memory_collection(client)
    client.load_collection("user_memories")
    dim = client.describe_collection("user_memories")
    for f in dim.get("fields", []):
        if f["name"] == "dense_embedding":
            print(f"Embedding dim: {f['params'].get('dim')}")

    # List all records by user_id
    print("\n=== All records grouped by user_id ===")
    # query a broad range — user_ids are likely > 0 or -1
    for uid in [-1, 0, 1, 2, 3]:
        rows = client.query(
            collection_name="user_memories",
            filter=f"user_id == {uid}",
            output_fields=["user_id", "memory_id", "content", "timestamp"],
            limit=100,
        )
        if rows:
            print(f"\nuser_id={uid} ({len(rows)} records):")
            for r in rows:
                print(f"  memory_id={r.get('memory_id')}  content={r.get('content')[:60]}")

    # Search query_text
    print(f"\n=== Search: '{query_text}' ===")
    vec = await get_dense_embedding(query_text)
    print(f"Embedding vector (first 5 dims): {vec[:5]}...")
    print(f"Vector length: {len(vec)}")

    results = search_memories(client, user_id=-1, dense_embedding=vec, top_k=10)
    print(f"\nuser_id=-1 results: {len(results)} hits")
    for r in results:
        print(f"  distance={r['distance']:.4f}  content={r['content'][:80]}")

    # Try with user_id=1
    results = search_memories(client, user_id=1, dense_embedding=vec, top_k=10)
    print(f"\nuser_id=1 results: {len(results)} hits")
    for r in results:
        print(f"  distance={r['distance']:.4f}  content={r['content'][:80]}")

    # Try without user_id filter (all users)
    print("\n=== Search without user_id filter ===")
    raw = client.search(
        collection_name="user_memories",
        data=[vec],
        anns_field="dense_embedding",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=10,
        output_fields=["user_id", "memory_id", "content"],
    )
    for group in raw:
        for hit in group:
            print(f"  id={hit['id']} distance={hit['distance']:.4f}  user_id={hit['entity'].get('user_id')}  content={hit['entity'].get('content')[:80]}")

    # Also check old collections
    for col in ["conversation_history", "knowledge_chunks"]:
        try:
            client.load_collection(col)
            schema = client.describe_collection(col)
            fields = [f["name"] for f in schema.get("fields", [])]
            print(f"\n{col} fields: {fields}")

            all_rows = client.query(
                collection_name=col,
                filter="",
                output_fields=fields,
                limit=200,
            )
            print(f"=== {col}: {len(all_rows)} records ===")
            for r in all_rows:
                content = r.get("content") or r.get("text", "") or ""
                role = r.get("role", "")
                uid = r.get("user_id", "")
                rid = r.get("memory_id") or r.get("message_id") or ""
                print(f"  role={role}  user_id={uid}  id={rid}  content={str(content)[:80]}")

            # Search query_text on conversation_history
            if col == "conversation_history" and "dense_embedding" in fields:
                print(f"\n>>> Search '{query_text}' in {col}:")
                raw = client.search(
                    collection_name=col,
                    data=[vec],
                    anns_field="dense_embedding",
                    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=10,
                    output_fields=["content", "role", "user_id"],
                )
                for group in raw:
                    for hit in group:
                        print(f"  distance={hit['distance']:.4f}  role={hit['entity'].get('role')}  content={str(hit['entity'].get('content', ''))[:80]}")
        except Exception as e:
            print(f"\n=== {col}: error — {e} ===")


if __name__ == "__main__":
    asyncio.run(main())
