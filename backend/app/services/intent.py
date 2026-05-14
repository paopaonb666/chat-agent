import json
import os
from pydantic import BaseModel
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
INTENT_MODEL = "qwen2.5:0.5b"

PROMPT_TEMPLATE = """你是一个对话意图分析助手。请分析用户输入，判断是否需要检索对话历史来回答。
输出严格为 JSON 格式，不要包含 markdown 代码块或其他说明：
{{
  "needs_retrieval": true/false,
  "refined_query": "用于检索的优化查询（如果需要检索）",
  "reason": "判断理由"
}}

当前对话历史（最近5轮）：
{history}

用户输入：{query}
"""


class IntentResult(BaseModel):
    needs_retrieval: bool = False
    refined_query: str = ""
    reason: str = ""


def _format_history(messages: list[dict]) -> str:
    recent = messages[-10:] if len(messages) > 10 else messages
    lines = []
    for m in recent:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "（无历史）"


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"needs_retrieval": False, "refined_query": "", "reason": "parse_failed"}


async def recognize_intent(query: str, messages: list[dict]) -> IntentResult:
    prompt = PROMPT_TEMPLATE.format(history=_format_history(messages), query=query)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": INTENT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        parsed = _extract_json(content)
        return IntentResult(**parsed)
