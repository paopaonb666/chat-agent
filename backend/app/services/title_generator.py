import asyncio
import logging

import httpx
from sqlalchemy.orm import Session

from app.models import Conversation

logger = logging.getLogger(__name__)


def _update_title(conv_id: str, title: str, db: Session) -> None:
    if not title:
        return
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if conv and conv.title == "新对话":
        conv.title = title[:30]
        db.commit()


async def auto_title_on_first_exchange(
    conv_id: str, first_msg: str, assistant_content: str, cfg: dict, db: Session
):
    title = None
    try:
        prompt = (
            f"为以下对话生成一个简洁的标题（3-8个字），直接返回标题文字，不要引号：\n"
            f"用户：{first_msg[:200]}\nAI：{assistant_content[:200]}"
        )
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{cfg['base_url']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg['api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 20,
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip().strip("\"'")
                if raw:
                    title = raw[:30]
    except Exception:
        logger.exception("Auto-title generation failed")

    await asyncio.to_thread(_update_title, conv_id, title or first_msg[:30], db)
