import logging
from app.services.embedding import get_dense_embedding
from app.core.milvus import get_milvus_client
from app.services.milvus_store import ensure_collection, insert_message

logger = logging.getLogger(__name__)


async def index_message(conv_id: str, user_id: int | None, role: str, content: str, msg_id: int):
    try:
        vec = await get_dense_embedding(content)
        client = get_milvus_client()
        ensure_collection(client)
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
        logger.exception("Milvus index failed")
