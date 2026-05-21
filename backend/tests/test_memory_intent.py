import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import httpx

from app.services.intent import should_extract_memory, PROMPT_TEMPLATE


def _fake_httpx_response(json_obj: dict):
    """Build a fake httpx response whose .json() returns OpenAI-compatible payload."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(json_obj, ensure_ascii=False)}}],
    }
    return resp


def _fake_httpx_str_response(text: str):
    """Build a fake httpx response returning non-JSON raw text."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
    }
    return resp


@pytest.mark.asyncio
async def test_should_extract_personal_name():
    messages = [
        {"role": "user", "content": "我叫炮炮，今年25岁，住在北京"},
        {"role": "assistant", "content": "很高兴认识你，炮炮！"},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": True,
        "memory_content": "用户叫炮炮，25岁，住在北京",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is True
    assert "炮炮" in content


@pytest.mark.asyncio
async def test_should_not_extract_operation():
    messages = [
        {"role": "user", "content": "帮我搜索一下北京的天气"},
        {"role": "assistant", "content": "好的，北京今天晴天，25度。"},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": False,
        "memory_content": "",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is False
    assert content == ""


@pytest.mark.asyncio
async def test_should_not_extract_greeting():
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": False,
        "memory_content": "",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is False


@pytest.mark.asyncio
async def test_should_extract_preference():
    messages = [
        {"role": "user", "content": "我很喜欢吃辣的，尤其是川菜和湘菜"},
        {"role": "assistant", "content": "好的，我会记住你喜欢川菜和湘菜。"},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": True,
        "memory_content": "用户喜欢吃辣，偏好川菜和湘菜",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is True
    assert "川菜" in content


@pytest.mark.asyncio
async def test_no_long_term_translation():
    messages = [
        {"role": "user", "content": "帮我把这段话翻译成英文"},
        {"role": "assistant", "content": "Here is the translation..."},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": False,
        "memory_content": "",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is False


@pytest.mark.asyncio
async def test_mixed_personal_and_task():
    messages = [
        {"role": "user", "content": "我叫小明，帮我写一封邮件给客户"},
        {"role": "assistant", "content": "好的小明，这是邮件草稿..."},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": True,
        "memory_content": "用户叫小明",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is True
    assert "小明" in content


@pytest.mark.asyncio
async def test_health_info():
    messages = [
        {"role": "user", "content": "我对青霉素过敏，之前有过严重过敏反应"},
        {"role": "assistant", "content": "了解了，我会注意。请务必告知医生。"},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": True,
        "memory_content": "用户对青霉素过敏，曾有严重过敏反应",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is True
    assert "过敏" in content


@pytest.mark.asyncio
async def test_fail_safe_on_connection_error():
    messages = [
        {"role": "user", "content": "我叫张三"},
        {"role": "assistant", "content": "你好张三"},
    ]
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        needs, content = await should_extract_memory(messages)
    assert needs is False
    assert content == ""


@pytest.mark.asyncio
async def test_fail_safe_on_http_error():
    messages = [
        {"role": "user", "content": "我叫张三"},
        {"role": "assistant", "content": "你好张三"},
    ]
    resp = MagicMock()
    resp.status_code = 500
    resp.text = "Internal Server Error"
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server error", request=MagicMock(), response=resp
    )
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=resp)
        needs, content = await should_extract_memory(messages)
    assert needs is False
    assert content == ""


@pytest.mark.asyncio
async def test_fail_safe_on_json_parse_error():
    messages = [
        {"role": "user", "content": "我叫张三"},
        {"role": "assistant", "content": "你好"},
    ]
    fake = _fake_httpx_str_response("这不是有效的 JSON，只是一段文本回复")
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is False
    assert content == ""


@pytest.mark.asyncio
async def test_empty_user_message():
    messages: list = []
    needs, content = await should_extract_memory(messages)
    assert needs is False
    assert content == ""


@pytest.mark.asyncio
async def test_should_not_extract_thanks():
    messages = [
        {"role": "user", "content": "谢谢你的帮助"},
        {"role": "assistant", "content": "不客气！"},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": False,
        "memory_content": "",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is False


@pytest.mark.asyncio
async def test_should_extract_career_goal():
    messages = [
        {"role": "user", "content": "我想在三年内成为技术总监，这是我的职业目标"},
        {"role": "assistant", "content": "很有志向！我可以帮你规划学习路线。"},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": True,
        "memory_content": "用户职业目标是三年内成为技术总监",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is True
    assert "技术总监" in content


@pytest.mark.asyncio
async def test_should_not_extract_code_review():
    messages = [
        {"role": "user", "content": "帮我 review 一下这段 Python 代码"},
        {"role": "assistant", "content": "这段代码有几个问题..."},
    ]
    fake = _fake_httpx_response({
        "needs_long_term_memory": False,
        "memory_content": "",
    })
    with patch("app.services.intent.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake)
        needs, content = await should_extract_memory(messages)
    assert needs is False


@pytest.mark.asyncio
async def test_prompt_template_format():
    """Ensure the prompt template renders correctly."""
    prompt = PROMPT_TEMPLATE.format(user_message="你好", assistant_message="你好！")
    assert "你好" in prompt
    assert "needs_long_term_memory" in prompt
    assert "memory_content" in prompt
    assert "用户身份信息" in prompt
    assert "操作指令" in prompt
