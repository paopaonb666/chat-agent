import json
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import asyncio
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import Conversation, Message
from app.services.rag_pipeline import run_rag
from app.services.embedding import get_dense_embedding
from app.services.milvus_store import get_milvus_client, ensure_collection, insert_message

router = APIRouter()

_milvus_client = None


def _get_milvus():
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = get_milvus_client()
        ensure_collection(_milvus_client)
    return _milvus_client


async def _index_message(conv_id: str, user_id: int | None, role: str, content: str, msg_id: int):
    try:
        vec = await get_dense_embedding(content)
        client = _get_milvus()
        insert_message(
            client,
            conversation_id=conv_id,
            user_id=user_id,
            role=role,
            content=content,
            message_id=msg_id,
            dense_embedding=vec,
        )
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Milvus index failed")

MODEL_CONFIG = {
    "deepseek-chat": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
    "glm-4": {
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
        "base_url": os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
        "model": os.getenv("ZHIPU_MODEL", "glm-4"),
    },
}


class CreateConversationRequest(BaseModel):
    title: str = "New Conversation"
    model: str = "deepseek-chat"


class ChatCompletionsRequest(BaseModel):
    conversation_id: str
    message: str = ""


def _save_assistant_message(db: Session, conv_id: str, content: str) -> None:
    db.add(Message(conversation_id=conv_id, role="assistant", content=content))
    db.commit()


@router.post("/conversations")
def create_conversation(payload: CreateConversationRequest, db: Session = Depends(get_db)):
    conv = Conversation(title=payload.title, model=payload.model)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"id": conv.id, "title": conv.title, "model": conv.model, "messages": []}


@router.get("/conversations")
def list_conversations(db: Session = Depends(get_db)):
    convs = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    return [
        {
            "id": c.id,
            "title": c.title,
            "model": c.model,
            "messages": [{"role": m.role, "content": m.content} for m in c.messages],
        }
        for c in convs
    ]


@router.get("/conversations/{conv_id}")
def get_conversation(conv_id: str, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": conv.id,
        "title": conv.title,
        "model": conv.model,
        "messages": [{"role": m.role, "content": m.content} for m in conv.messages],
    }


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionsRequest, db: Session = Depends(get_db)):
    conv_id = payload.conversation_id
    message = payload.message
    if not conv_id:
        raise HTTPException(status_code=400, detail="conversation_id required")

    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    new_msg = Message(conversation_id=conv_id, role="user", content=message)
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    db.refresh(conv)
    messages = [{"role": m.role, "content": m.content} for m in conv.messages]

    asyncio.create_task(
        _index_message(conv_id, conv.user_id, "user", message, new_msg.id)
    )

    rag_context = await run_rag(message, conv_id, conv.user_id, messages)
    if rag_context:
        messages = [{"role": "system", "content": rag_context}] + messages

    cfg = MODEL_CONFIG.get(conv.model, MODEL_CONFIG["deepseek-chat"])
    api_key = cfg["api_key"]
    base_url = cfg["base_url"]
    model_name = cfg["model"]

    async def event_stream():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model_name, "messages": messages, "stream": True},
                timeout=60.0,
            ) as response:
                if response.status_code != 200:
                    _ = await response.aread()
                    yield f"data: {json.dumps({'error': f'API returned {response.status_code}'})}\n\n"
                    return

                assistant_content = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            await asyncio.to_thread(
                                _save_assistant_message, db, conv_id, assistant_content
                            )
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(data)
                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            assistant_content += content
                            yield f"data: {json.dumps({'content': content})}\n\n"
                            if choice.get("finish_reason") is not None:
                                await asyncio.to_thread(
                                    _save_assistant_message, db, conv_id, assistant_content
                                )
                                yield "data: [DONE]\n\n"
                                break
                        except Exception:
                            continue

    return StreamingResponse(event_stream(), media_type="text/event-stream")
