import uuid
import contextvars

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "user_id", default="-"
)
conversation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "conversation_id", default="-"
)


def generate_request_id() -> str:
    return str(uuid.uuid4())
