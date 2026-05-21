class ChatAgentException(Exception):
    """Base exception for business logic errors."""
    pass


class MemoryNotFoundException(ChatAgentException):
    pass


class ExternalServiceException(ChatAgentException):
    """LLM / embedding / search service failure."""
    pass


class VectorStoreException(ChatAgentException):
    """Milvus / vector DB failure."""
    pass
