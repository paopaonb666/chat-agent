import asyncio
import os
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RERANK_MODEL = os.getenv("RERANK_MODEL", "pdurugyan/qwen3-reranker-0.6b-q8_0")

PROMPT_TEMPLATE = """Passage: {passage}
Query: {query}
Determine if the passage is relevant to the query. Output a relevance score from 0 to 10, where 0 means completely irrelevant and 10 means highly relevant. Only output the number.
Score:"""


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


async def rerank_passages(query: str, passages: list[dict], top_n: int = 5) -> list[dict]:
    if not passages:
        return []
    prompts = [_build_prompt(query, p["content"]) for p in passages]
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": RERANK_MODEL,
                    "prompt": p,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10},
                },
                timeout=30.0,
            )
            for p in prompts
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    scored = []
    for i, resp in enumerate(responses):
        score = 0.0
        if isinstance(resp, httpx.Response):
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            score = _extract_score(text)
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [passages[i] for i, _ in scored[:top_n]]
