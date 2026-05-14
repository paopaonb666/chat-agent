import os
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


async def get_dense_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], (int, float)):
            return embeddings
        raise ValueError(f"Unexpected embedding response format: {data}")


async def get_embedding_dim() -> int:
    vec = await get_dense_embedding("test")
    return len(vec)
