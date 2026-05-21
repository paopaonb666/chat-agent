import httpx
from app.core.config import settings


async def get_dense_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.embedding_model, "input": text},
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
