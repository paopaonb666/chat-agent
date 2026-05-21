import pytest
from unittest.mock import patch, MagicMock
import app.services.memory_client as mc


@pytest.fixture(autouse=True)
def reset_singleton():
    mc._memory_instance = None


def test_get_memory_returns_singleton():
    mock_memory = MagicMock()
    with patch("app.services.memory_client.Memory") as MockMemory:
        MockMemory.from_config.return_value = mock_memory
        m1 = mc.get_memory()
        m2 = mc.get_memory()
        assert m1 is m2
        MockMemory.from_config.assert_called_once()


def test_memory_config_uses_settings():
    mock_memory = MagicMock()
    with patch("app.services.memory_client.Memory") as MockMemory:
        MockMemory.from_config.return_value = mock_memory
        mc.get_memory()
        config = MockMemory.from_config.call_args[0][0]

        assert config["llm"]["provider"] == "deepseek"
        assert config["embedder"]["provider"] == "ollama"
        assert config["vector_store"]["provider"] == "milvus"
        assert config["vector_store"]["config"]["collection_name"] == "mem0_memories"
        assert config["vector_store"]["config"]["metric_type"] == "COSINE"
        assert "custom_instructions" in config
        assert "中文" in config["custom_instructions"]


def test_get_memory_does_not_recreate():
    mock_memory = MagicMock()
    with patch("app.services.memory_client.Memory") as MockMemory:
        MockMemory.from_config.return_value = mock_memory
        mc.get_memory()
        MockMemory.from_config.reset_mock()
        mc.get_memory()
        MockMemory.from_config.assert_not_called()
