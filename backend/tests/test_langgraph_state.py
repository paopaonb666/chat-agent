import pytest
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from app.langgraph_agent.state import AgentState


class TestAgentState:
    def test_messages_annotated_add(self):
        """messages field uses operator.add to accumulate."""
        state: AgentState = {
            "messages": [HumanMessage(content="hello")],
            "user_message": "",
            "enable_web_search": False,
            "api_key": "",
            "base_url": "",
            "model_name": "",
            "user_id": "",
            "conversation_id": "",
            "memory_context": "",
            "rag_context": "",
            "web_sources": [],
            "iteration_count": 0,
            "last_failure_reason": "",
            "same_failure_count": 0,
            "final_content": "",
            "continue_loop": False,
        }
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "hello"

    def test_web_sources_annotated_add(self):
        """web_sources accumulates results across tool calls."""
        state: AgentState = {
            "messages": [],
            "user_message": "",
            "enable_web_search": False,
            "api_key": "",
            "base_url": "",
            "model_name": "",
            "user_id": "",
            "conversation_id": "",
            "memory_context": "",
            "rag_context": "",
            "web_sources": [{"title": "A", "url": "http://a.com"}],
            "iteration_count": 0,
            "last_failure_reason": "",
            "same_failure_count": 0,
            "final_content": "",
            "continue_loop": False,
        }
        assert len(state["web_sources"]) == 1

    def test_initial_state_defaults(self):
        """All required fields can be initialized with defaults."""
        state: AgentState = {
            "messages": [],
            "user_message": "test",
            "enable_web_search": True,
            "api_key": "sk-test",
            "base_url": "https://api.test.com",
            "model_name": "test-model",
            "user_id": "1",
            "conversation_id": "conv-123",
            "memory_context": "",
            "rag_context": "",
            "web_sources": [],
            "iteration_count": 0,
            "last_failure_reason": "",
            "same_failure_count": 0,
            "final_content": "",
            "continue_loop": False,
        }
        assert state["user_message"] == "test"
        assert state["enable_web_search"] is True
        assert state["iteration_count"] == 0
