import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.arq import get_arq_pool
from app.core.config import settings
from app.core.milvus import get_milvus_client
from app.db import get_db
from app.deps import get_admin_user, get_current_user
from app.models import KnowledgeDocument, KnowledgeChunk, User
from app.services.knowledge_milvus import delete_by_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])

KB_UPLOADS_ROOT = Path("uploads/knowledge").resolve()
KB_MAX_FILE_SIZE = settings.knowledge_upload_max_size

KB_ALLOWED_MIME_TYPES = {
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/markdown",
    "text/html",
    "text/csv",
}


def _build_document_list_query(db: Session, user: User):
    q = db.query(KnowledgeDocument)
    if user.role != "admin":
        q = q.filter(KnowledgeDocument.visibility.in_(["public", "shared"]))
    return q


# ── Request / Response models ──────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    status: str


class DocumentItem(BaseModel):
    id: str
    filename: str
    mime_type: str
    file_size: int
    visibility: str
    status: str
    chunk_count: int
    created_at: str

    @classmethod
    def from_doc(cls, d: KnowledgeDocument) -> "DocumentItem":
        return cls(
            id=d.id,
            filename=d.filename,
            mime_type=d.mime_type,
            file_size=d.file_size,
            visibility=d.visibility,
            status=d.status,
            chunk_count=d.chunk_count,
            created_at=d.created_at.isoformat(),
        )


class ChunkItem(BaseModel):
    id: str
    content: str
    chunk_index: int
    title_path: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    title_path: str
    distance: float


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    visibility: str = Form("public"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user),
):
    mime_type = file.content_type or "application/octet-stream"
    if mime_type not in KB_ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{mime_type}' not allowed",
        )

    content = await file.read()
    if len(content) > KB_MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content)} bytes). Maximum: {KB_MAX_FILE_SIZE} bytes",
        )

    doc_id = str(uuid.uuid4())
    original_name = file.filename or "untitled"
    safe_name = f"{uuid.uuid4()}{Path(original_name).suffix}"
    upload_dir = KB_UPLOADS_ROOT / doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = (upload_dir / safe_name).resolve()
    if not str(file_path).startswith(str(KB_UPLOADS_ROOT)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    with open(file_path, "wb") as f:
        f.write(content)

    doc = KnowledgeDocument(
        id=doc_id,
        filename=original_name,
        mime_type=mime_type,
        file_size=len(content),
        owner_id=current_user.id,
        visibility=visibility if visibility in ("private", "shared", "public") else "public",
        status="pending",
        path=str(file_path),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    pool = await get_arq_pool()
    await pool.enqueue_job("process_document_task", doc_id)

    return {"id": doc.id, "filename": doc.filename, "status": doc.status}


@router.get("/documents")
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = _build_document_list_query(db, current_user)
    if status:
        q = q.filter(KnowledgeDocument.status == status)
    total = q.count()
    docs = q.order_by(KnowledgeDocument.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {
        "items": [DocumentItem.from_doc(d) for d in docs],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/documents/{doc_id}")
def get_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = _build_document_list_query(db, current_user)
    doc = q.filter(KnowledgeDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentItem.from_doc(doc)


@router.get("/documents/{doc_id}/chunks")
def get_document_chunks(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = _build_document_list_query(db, current_user)
    doc = q.filter(KnowledgeDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = (
        db.query(KnowledgeChunk)
        .filter(KnowledgeChunk.document_id == doc_id)
        .order_by(KnowledgeChunk.chunk_index)
        .all()
    )
    return [
        {
            "id": c.id,
            "content": c.content,
            "chunk_index": c.chunk_index,
            "title_path": c.title_path,
        }
        for c in chunks
    ]


@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user),
):
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete Milvus entries
    try:
        client = get_milvus_client()
        delete_by_document(client, doc_id)
    except Exception:
        logger.exception("Failed to delete Milvus entries for %s", doc_id)

    # Delete file
    try:
        upload_dir = KB_UPLOADS_ROOT / doc_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
    except Exception:
        logger.exception("Failed to delete file for %s", doc_id)

    db.delete(doc)
    db.commit()
    return {"detail": "Document deleted"}


@router.post("/documents/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user),
):
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete old chunks and Milvus data
    db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc_id).delete()
    try:
        client = get_milvus_client()
        delete_by_document(client, doc_id)
    except Exception:
        logger.exception("Failed to delete old Milvus entries for %s", doc_id)

    doc.status = "pending"
    doc.chunk_count = 0
    doc.vector_indexed = False
    doc.error_message = ""
    db.commit()

    pool = await get_arq_pool()
    await pool.enqueue_job("process_document_task", doc_id)

    return {"detail": "Reprocessing started", "id": doc_id, "status": "pending"}


@router.get("/documents/{doc_id}/status")
def get_document_status(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = _build_document_list_query(db, current_user)
    doc = q.filter(KnowledgeDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"id": doc.id, "status": doc.status, "error_message": doc.error_message}


@router.post("/search")
async def search_knowledge(
    req: SearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.services.embedding import get_dense_embedding
    from app.services.knowledge_milvus import search_knowledge_base

    dense_vector = await get_dense_embedding(req.query)
    client = get_milvus_client()

    q = _build_document_list_query(db, current_user)
    visible_docs = q.filter(
        KnowledgeDocument.status == "completed",
        KnowledgeDocument.vector_indexed.is_(True),
    ).all()
    if not visible_docs:
        return []
    document_ids = [d.id for d in visible_docs]

    results = search_knowledge_base(client, dense_vector, document_ids, top_k=req.top_k)
    return [
        {
            "chunk_id": r["id"],
            "document_id": r["document_id"],
            "content": r["content"],
            "distance": r["distance"],
        }
        for r in results
    ]
