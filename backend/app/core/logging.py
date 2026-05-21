import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "conversation_id"):
            log_entry["conversation_id"] = record.conversation_id
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        from app.core.context import request_id_var, user_id_var, conversation_id_var

        record.request_id = request_id_var.get("-")
        record.user_id = user_id_var.get("-")
        record.conversation_id = conversation_id_var.get("-")
        return True


def setup_logging(
    level: int = logging.INFO, json_format: bool = False, log_file: str = ""
) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers to avoid duplicates on reload
    root.handlers.clear()

    # Context filter for request/user/conversation IDs
    root.addFilter(ContextFilter())

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    if json_format:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )
    root.addHandler(console)

    # File handler with rotation
    if log_file:
        from app.core.config import settings

        max_bytes = getattr(settings, "log_max_bytes", 10 * 1024 * 1024)
        backup_count = getattr(settings, "log_backup_count", 5)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                )
            )
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
