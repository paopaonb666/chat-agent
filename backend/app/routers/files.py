import io
import logging
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import Conversation, UploadedFile
from app.deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/files", tags=["files"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
UPLOADS_ROOT = Path("uploads").resolve()


def _extract_text(content: bytes, mime_type: str) -> str:
    if mime_type == "text/plain":
        return content.decode("utf-8")
    if mime_type == "application/pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            logger.exception("PDF text extraction failed")
            return ""
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            logger.exception("DOCX text extraction failed")
            return ""
    return ""


@router.post("/upload", summary="上传文件", description="上传文件到指定对话。支持 txt/pdf/docx 格式，最大 10MB。", responses={400: {"description": "文件类型不支持或文件过大"}, 404: {"description": "对话不存在"}})
async def upload_file(
    conversation_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Validate file type
    mime_type = file.content_type or "application/octet-stream"
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{mime_type}' not allowed. Supported: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
        )

    # Read content with size check
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content)} bytes). Maximum: {MAX_FILE_SIZE} bytes",
        )

    # Safe filename
    original_name = file.filename or "untitled"
    safe_name = f"{uuid.uuid4()}{Path(original_name).suffix}"

    # Path traversal protection
    upload_dir = UPLOADS_ROOT / conversation_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = (upload_dir / safe_name).resolve()
    if not str(file_path).startswith(str(UPLOADS_ROOT)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    with open(file_path, "wb") as f:
        f.write(content)

    extracted_text = _extract_text(content, mime_type)

    db_file = UploadedFile(
        conversation_id=conversation_id,
        filename=original_name,
        path=str(file_path),
        mime_type=mime_type,
        extracted_text=extracted_text,
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return {
        "id": db_file.id,
        "filename": db_file.filename,
        "mime_type": db_file.mime_type,
        "extracted_text": db_file.extracted_text,
    }
