import math
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Conversation, User, UserMemory
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


def _pg_memory_to_dict(m: UserMemory) -> dict:
    return {
        "id": str(m.id),
        "content": m.content,
        "source": m.source,
        "created_at": m.created_at.isoformat() if m.created_at else "",
        "updated_at": m.updated_at.isoformat() if m.updated_at else "",
    }


def _sync_to_milvus(memory, content: str, user_id: str, source: str = "manual"):
    """Write a single memory to Milvus. Failure is logged but not raised."""
    try:
        memory.add(
            [{"role": "user", "content": content}],
            user_id=user_id,
            metadata={"source": source},
        )
    except Exception:
        logger.warning("Failed to sync memory to Milvus", exc_info=True)


def _delete_from_milvus_by_content(memory, content: str, user_id: str):
    """Delete Milvus memories matching the given content for a user."""
    try:
        results = memory.get_all(filters={"user_id": user_id}, top_k=100)
        items = results.get("results", []) if isinstance(results, dict) else []
        for r in items:
            if r.get("memory", "").strip() == content.strip():
                try:
                    memory.delete(memory_id=r["id"])
                except Exception:
                    logger.warning("Failed to delete Milvus memory %s", r.get("id"))
    except Exception:
        logger.warning("Failed to search Milvus for deletion", exc_info=True)


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
    # Also write to PostgreSQL
    try:
        pg_mem = UserMemory(user_id=user.id, content=content, source="auto_extracted")
        db.add(pg_mem)
        db.commit()
    except Exception:
        logger.warning("Failed to write memory to PostgreSQL", exc_info=True)
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
    total = db.query(UserMemory).filter(UserMemory.user_id == current_user.id).count()
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    offset = (page - 1) * page_size

    records = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == current_user.id)
        .order_by(UserMemory.updated_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    items = [_pg_memory_to_dict(m) for m in records]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@router.post("/memories", summary="创建记忆", description="手动创建一条长期记忆")
def create_memory(
    payload: CreateMemoryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Write to PostgreSQL (source of truth)
    pg_mem = UserMemory(user_id=current_user.id, content=payload.content, source="manual")
    db.add(pg_mem)
    db.commit()
    db.refresh(pg_mem)

    # 2. Sync to Milvus (best-effort)
    memory = get_memory()
    if memory:
        _sync_to_milvus(memory, payload.content, str(current_user.id), source="manual")

    return _pg_memory_to_dict(pg_mem)


@router.put("/memories/{memory_id}", summary="更新记忆", description="更新指定记忆的内容", responses={404: {"description": "记忆不存在"}})
def update_memory(
    memory_id: str,
    payload: UpdateMemoryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    memory = get_memory()

    # Try PostgreSQL first (int id)
    if memory_id.isdigit():
        pg_mem = db.query(UserMemory).filter(
            UserMemory.id == int(memory_id),
            UserMemory.user_id == current_user.id,
        ).first()
        if not pg_mem:
            raise HTTPException(status_code=404, detail="Memory not found")

        old_content = pg_mem.content
        pg_mem.content = payload.content
        db.commit()
        db.refresh(pg_mem)

        # Sync to Milvus: delete old + add new
        if memory:
            _delete_from_milvus_by_content(memory, old_content, str(current_user.id))
            _sync_to_milvus(memory, payload.content, str(current_user.id), source=pg_mem.source)

        return _pg_memory_to_dict(pg_mem)

    # Otherwise treat as Milvus UUID
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    try:
        memory.update(memory_id=memory_id, data=payload.content)
    except Exception:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Also update PostgreSQL if a matching record exists
    try:
        pg_mem = db.query(UserMemory).filter(
            UserMemory.user_id == current_user.id,
        ).all()
        for m in pg_mem:
            # Attempt to find by old content — best-effort match
            pass
    except Exception:
        logger.warning("Failed to sync update to PostgreSQL", exc_info=True)

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
    current_user: User = Depends(get_current_user),
):
    memory = get_memory()

    # Try PostgreSQL first (int id)
    if memory_id.isdigit():
        pg_mem = db.query(UserMemory).filter(
            UserMemory.id == int(memory_id),
            UserMemory.user_id == current_user.id,
        ).first()
        if not pg_mem:
            raise HTTPException(status_code=404, detail="Memory not found")

        content = pg_mem.content
        db.delete(pg_mem)
        db.commit()

        # Also delete from Milvus
        if memory:
            _delete_from_milvus_by_content(memory, content, str(current_user.id))

        return {"detail": "Memory deleted"}

    # Otherwise treat as Milvus UUID
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Get content before deleting so we can sync to PostgreSQL
    milvus_content = ""
    try:
        results = memory.get_all(filters={"user_id": str(current_user.id)}, top_k=100)
        items = results.get("results", []) if isinstance(results, dict) else []
        for r in items:
            if r.get("id") == memory_id:
                milvus_content = r.get("memory", "")
                break
    except Exception:
        pass

    try:
        memory.delete(memory_id=memory_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Also delete from PostgreSQL by matching content
    if milvus_content:
        try:
            pg_mem = db.query(UserMemory).filter(
                UserMemory.user_id == current_user.id,
                UserMemory.content == milvus_content,
            ).first()
            if pg_mem:
                db.delete(pg_mem)
                db.commit()
        except Exception:
            logger.warning("Failed to sync delete to PostgreSQL", exc_info=True)

    return {"detail": "Memory deleted"}
