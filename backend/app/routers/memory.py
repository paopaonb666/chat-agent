import math
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Conversation, User
from app.deps import get_current_user
from app.services.memory_client import get_memory

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memory"])


class StoreMemoryRequest(BaseModel):
    content: str
    conversation_id: str


class CreateMemoryRequest(BaseModel):
    content: str


class UpdateMemoryRequest(BaseModel):
    content: str


def _enrich_results(results, default_source="auto_extracted"):
    """Normalize mem0 result dicts to our API response format."""
    items = results if isinstance(results, list) else results.get("results", [])
    return [
        {
            "id": r.get("id", ""),
            "content": r.get("memory", r.get("content", "")),
            "source": r.get("metadata", {}).get("source", default_source) if isinstance(r.get("metadata"), dict) else default_source,
            "created_at": r.get("created_at", ""),
            "updated_at": r.get("updated_at", ""),
        }
        for r in items
    ]


def _paginate(items: list, page: int, page_size: int) -> dict:
    total = len(items)
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    offset = (page - 1) * page_size
    return {
        "items": items[offset:offset + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


async def _store_memory_steps(content: str, conv_id: str, db: Session, user: User):
    yield json.dumps({"type": "step", "step": "prepare", "status": "running", "label": "准备分析"}, ensure_ascii=False)
    yield json.dumps({"type": "step", "step": "prepare", "status": "completed", "label": "准备分析"}, ensure_ascii=False)

    user_id = str(user.id)

    yield json.dumps({"type": "step", "step": "embed", "status": "running", "label": "生成向量"}, ensure_ascii=False)
    yield json.dumps({"type": "step", "step": "embed", "status": "completed", "label": "生成向量"}, ensure_ascii=False)

    yield json.dumps({"type": "step", "step": "dedup", "status": "running", "label": "去重检查"}, ensure_ascii=False)
    yield json.dumps({"type": "step", "step": "dedup", "status": "completed", "label": "去重检查"}, ensure_ascii=False)

    yield json.dumps({"type": "step", "step": "save", "status": "running", "label": "存入数据库"}, ensure_ascii=False)
    memory = get_memory()
    memory.add(
        [{"role": "user", "content": content}],
        user_id=user_id,
        metadata={"source": "auto_extracted"},
    )
    yield json.dumps({"type": "step", "step": "save", "status": "completed", "label": "存入数据库"}, ensure_ascii=False)

    yield json.dumps({"type": "step", "step": "index", "status": "running", "label": "存入向量库"}, ensure_ascii=False)
    yield json.dumps({"type": "step", "step": "index", "status": "completed", "label": "存入向量库"}, ensure_ascii=False)

    yield json.dumps({"type": "done", "message": "长期记忆已存储"}, ensure_ascii=False)


@router.post("/memory/store", summary="存储记忆（流式）", description="分析对话并自动存储长期记忆，返回 SSE 步骤事件流")
async def store_memory(payload: StoreMemoryRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    async def event_stream():
        async for event in _store_memory_steps(payload.content, payload.conversation_id, db, current_user):
            yield f"data: {event}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── CRUD endpoints ────────────────────────────────────────────────────────


@router.get("/memories/search", summary="搜索记忆", description="根据关键词搜索当前用户的长期记忆，支持分页")
def search_memories_endpoint(
    q: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    top_k: int = 10,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if not q.strip():
        return {"items": [], "total": 0, "page": 1, "page_size": page_size, "total_pages": 0}

    try:
        user = current_user
        memory = get_memory()
        results = memory.search(
            query=q,
            filters={"user_id": str(user.id)},
            top_k=top_k,
        )
        items = results.get("results", []) if isinstance(results, dict) else []
    except Exception:
        logger.exception("Memory search failed")
        return {"items": [], "total": 0, "page": 1, "page_size": page_size, "total_pages": 0}

    enriched = []
    for r in items:
        enriched.append({
            "id": r.get("id", ""),
            "content": r.get("memory", ""),
            "distance": round(r.get("score", 0), 4),
            "source": "auto_extracted",
            "created_at": r.get("created_at", ""),
            "updated_at": r.get("updated_at", ""),
        })
    return _paginate(enriched, page, page_size)


@router.get("/memories", summary="获取记忆列表", description="获取当前用户的所有长期记忆，按时间降序，支持分页")
def list_memories(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    try:
        user = current_user
        memory = get_memory()
        results = memory.get_all(filters={"user_id": str(user.id)})
        all_items = _enrich_results(results)
    except Exception:
        logger.exception("Memory list failed")
        return {"items": [], "total": 0, "page": 1, "page_size": page_size, "total_pages": 0}
    return _paginate(all_items, page, page_size)


@router.post("/memories", summary="创建记忆", description="手动创建一条长期记忆")
def create_memory(
    payload: CreateMemoryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = current_user
    memory = get_memory()
    result = memory.add(
        [{"role": "user", "content": payload.content}],
        user_id=str(user.id),
        metadata={"source": "manual"},
    )
    added = result.get("results", []) if isinstance(result, dict) else []
    if added:
        r = added[0]
        return {
            "id": r.get("id", ""),
            "content": r.get("memory", payload.content),
            "source": "manual",
            "created_at": r.get("created_at", ""),
            "updated_at": r.get("updated_at", ""),
        }
    return {
        "id": "", "content": payload.content, "source": "manual",
        "created_at": "", "updated_at": "",
    }


@router.put("/memories/{memory_id}", summary="更新记忆", description="更新指定记忆的内容", responses={404: {"description": "记忆不存在"}})
def update_memory(
    memory_id: str,
    payload: UpdateMemoryRequest,
    db: Session = Depends(get_db),
):
    memory = get_memory()
    try:
        memory.update(memory_id=memory_id, data=payload.content)
    except Exception:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {
        "id": memory_id,
        "content": payload.content,
        "source": "",
        "created_at": "",
        "updated_at": "",
    }


@router.delete("/memories/{memory_id}", summary="删除记忆", description="删除指定记忆", responses={404: {"description": "记忆不存在"}})
def delete_memory_endpoint(
    memory_id: str,
    db: Session = Depends(get_db),
):
    memory = get_memory()
    try:
        memory.delete(memory_id=memory_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"detail": "Memory deleted"}
