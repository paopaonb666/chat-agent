from typing import Any
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./chat_agent.db"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # Zhipu
    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    zhipu_model: str = "glm-4"

    # Milvus
    milvus_uri: str = "./milvus.db"

    # Hybrid search time decay
    time_decay_max_bonus: float = 0.2
    time_decay_window_size: int = 200

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    rerank_model: str = "pdurugyan/qwen3-reranker-0.6b-q8_0"
    ollama_large_model: str = "qwen2.5:1.5b"

    # Tavily
    tavily_api_key: str = ""

    # Search engine priority (comma-separated, e.g. "tavily,duckduckgo")
    search_engine_priority: str = "tavily,duckduckgo"

    # Auth
    secret_key: str = "dev-secret-change-in-production"

    # Logging
    log_file: str = ""
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_backup_count: int = 5

    # Environment
    env: str = "development"

    # CORS — use Any to avoid Pydantic Settings json-decoding env strings
    cors_origins: Any = "http://localhost:5173"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("milvus_uri", mode="before")
    @classmethod
    def _validate_milvus_uri(cls, v: str) -> str:
        if isinstance(v, str) and v and not (
            v.startswith(("http://", "https://")) or v.endswith(".db")
        ):
            raise ValueError("MILVUS_URI must start with http://, https://, or end with .db")
        return v

    @field_validator("env", mode="before")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        if isinstance(v, str) and v not in ("development", "production", "testing"):
            raise ValueError("ENV must be one of: development, production, testing")
        return v

    @field_validator("time_decay_max_bonus", mode="before")
    @classmethod
    def _validate_time_decay_max_bonus(cls, v: float) -> float:
        if isinstance(v, (int, float)) and not (0.0 <= float(v) <= 1.0):
            raise ValueError("TIME_DECAY_MAX_BONUS must be between 0.0 and 1.0")
        return v

    # SiliconFlow (used for memory intent recognition)
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Memory intent filter
    enable_memory_intent_filter: bool = True

    # LangGraph agent feature flag
    use_langgraph_agent: bool = False

    # Redis / ARQ
    redis_url: str = "redis://localhost:6379"

    # Knowledge base
    knowledge_upload_max_size: int = 10 * 1024 * 1024  # 10 MB
    knowledge_chunk_max_chars: int = 800
    knowledge_chunk_overlap: int = 0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
