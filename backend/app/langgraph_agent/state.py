from typing import Annotated, TypedDict
import operator
from langchain.messages import AnyMessage


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_message: str
    enable_web_search: bool
    api_key: str
    base_url: str
    model_name: str
    user_id: str
    conversation_id: str
    memory_context: str
    rag_context: str
    web_sources: Annotated[list[dict], operator.add]
    iteration_count: int
    last_failure_reason: str
    same_failure_count: int
    final_content: str
    continue_loop: bool
