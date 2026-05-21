import asyncio
import httpx
from app.core.config import settings

PROMPT_TEMPLATE = """Passage: {passage}
Query: {query}
Determine if the passage is relevant to the query. Output a relevance score from 0 to 10, where 0 means completely irrelevant and 10 means highly relevant. Only output the number.
Score:"""

# Limit concurrent Ollama requests to avoid overwhelming the 1.5B model
_MAX_CONCURRENCY = 4


def _build_prompt(query: str, passage: str) -> str:
    return PROMPT_TEMPLATE.format(query=query, passage=passage)


def _extract_score(text: str) -> float:
    text = text.strip()
    for token in text.split():
        try:
            return float(token)
        except ValueError:
            continue
    return 0.0


async def _score_one(client: httpx.AsyncClient, prompt: str, sem: asyncio.Semaphore) -> httpx.Response:
    async with sem:
        return await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_large_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10},
            },
            timeout=30.0,
        )


async def rerank_passages(query: str, passages: list[dict], top_n: int = 5) -> list[dict]:
    if not passages:
        return []

    prompts = [_build_prompt(query, p["content"]) for p in passages]
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    async with httpx.AsyncClient() as client:
        tasks = [_score_one(client, p, sem) for p in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    scored = []
    for i, resp in enumerate(responses):
        score = 0.0
        if isinstance(resp, httpx.Response):
            try:
                resp.raise_for_status()
                text = resp.json().get("message", {}).get("content", "").strip()
                score = _extract_score(text)
            except httpx.HTTPStatusError:
                pass
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [passages[i] for i, _ in scored[:top_n]]
