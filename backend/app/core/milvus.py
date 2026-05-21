import logging
import threading
from pymilvus import MilvusClient
from app.core.config import settings

logger = logging.getLogger(__name__)
_milvus_client: MilvusClient | None = None
_lock = threading.Lock()


def get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        with _lock:
            if _milvus_client is None:
                _milvus_client = MilvusClient(uri=settings.milvus_uri)
                logger.debug("Milvus client initialized: %s", settings.milvus_uri)
    return _milvus_client
