import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.arq import WorkerSettings
from app.core.config import settings
from app.core.milvus import get_milvus_client
from app.db import SessionLocal
from app.models import KnowledgeDocument, KnowledgeChunk
from app.services.document_parser import parse_document, DocumentParseException
from app.services.embedding import get_dense_embedding
from app.services.knowledge_milvus import ensure_kb_collection, insert_knowledge_chunks
from app.services.semantic_chunker import chunk_document

logger = logging.getLogger(__name__)


@WorkerSettings.register_function
async def process_document_task(ctx, document_id: str):
    db = SessionLocal()
    try:
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()
        if not doc:
            logger.warning("Document %s not found for processing", document_id)
            return

        doc.status = "processing"
        db.commit()

        file_path = Path(doc.path)
        content = file_path.read_bytes()

        parsed = parse_document(content, doc.mime_type, doc.filename)
        chunks = chunk_document(parsed, max_chunk_chars=settings.knowledge_chunk_max_chars)

        chunk_records = []
        for i, chunk in enumerate(chunks):
            record = KnowledgeChunk(
                document_id=doc.id,
                content=chunk["content"],
                chunk_index=chunk["chunk_index"],
                title_path=chunk["title_path"],
                meta_json=json.dumps({"level": chunk.get("level", 0)}),
            )
            db.add(record)
            chunk_records.append(record)
        db.commit()
        for r in chunk_records:
            db.refresh(r)

        client = get_milvus_client()
        ensure_kb_collection(client)

        milvus_data = []
        for record in chunk_records:
            embedding = await get_dense_embedding(record.content)
            milvus_data.append({
                "id": record.id,
                "embedding": embedding,
                "document_id": doc.id,
                "content": record.content,
                "meta_json": record.meta_json,
                "user_id": doc.owner_id,
            })

        insert_knowledge_chunks(client, milvus_data)

        doc.status = "completed"
        doc.chunk_count = len(chunks)
        doc.vector_indexed = True
        doc.processed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Document %s processed: %s chunks", document_id, len(chunks))

    except DocumentParseException as e:
        logger.exception("Document parse failed for %s", document_id)
        doc.status = "failed"
        doc.error_message = str(e)[:500]
        db.commit()
    except Exception:
        logger.exception("Document processing failed for %s", document_id)
        doc.status = "failed"
        doc.error_message = "Internal processing error"
        db.commit()
        raise
    finally:
        db.close()
