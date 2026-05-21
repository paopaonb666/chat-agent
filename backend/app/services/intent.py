"""
记忆意图识别 — 在mem0提取之前过滤对话。
使用SiliconFlow的Qwen2.5-7B来判断对话回合是否包含值得长期记忆的
个人信息，避免操作噪音污染记忆存储。
"""
import json
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """你是一个记忆筛选助手。你的任务是判断一段对话是否包含值得长期记忆的用户个人信息，并从对话中提取出来。

## 需要长期记忆（needs_long_term_memory=true）

这些信息应在 memory_content 中提取出来：

- 用户身份信息：姓名、年龄、性别、职业、学历、公司
- 用户联系方式：电话、邮箱、社交账号
- 用户地理位置：居住城市、工作地址、常去地点
- 用户个人经历：工作经历、学习经历、人生重要事件
- 用户偏好习惯：饮食偏好、兴趣爱好、作息习惯、消费偏好、阅读偏好
- 用户情感与人际关系：家庭情况、伴侣/朋友信息、情感状态
- 用户健康状况：过敏史、慢性病、用药情况（用户主动提及的）
- 用户的长期目标、价值观、人生规划、信仰

## 不需要长期记忆（needs_long_term_memory=false）

这些内容应跳过，memory_content 填空字符串：

- 用户在软件上的操作指令：搜索、打开设置、生成图片、翻译等
- 临时性任务请求：帮我写邮件、查天气、写代码、改bug
- 与 AI 的交互行为：你好、谢谢、再见、你说得对
- 通用知识问答（不涉及用户个人信息）
- 当前会话上下文（如讨论的某个话题，除非用户明确表达了个人偏好）
- 用户对 AI 回答的评价或反馈

## 输出格式

严格输出 JSON，只包含以下两个字段：
{{
  "needs_long_term_memory": true/false,
  "memory_content": "提取出的个人信息摘要，用中文描述。如果不需要记忆则为空字符串。"
}}

## 对话内容

用户：{user_message}

AI：{assistant_message}"""


async def should_extract_memory(messages: list[dict]) -> tuple[bool, str]:
    """检查当前对话回合是否包含值得记忆的个人信息。

    参数:
        messages: 本轮对话的消息列表，格式为 {"role": "user"/"assistant", "content": "..."}

    返回:
        (needs_long_term_memory, memory_content) — memory_content 是适合直接存储到mem0的中文摘要。
        任何错误情况下返回 (False, "")（故障安全）。
    """
    user_message = ""
    assistant_message = ""
    for m in messages:
        if m.get("role") == "user":
            user_message = m.get("content", "")
        elif m.get("role") == "assistant":
            assistant_message = m.get("content", "")

    if not user_message.strip():
        return False, ""

    prompt = PROMPT_TEMPLATE.format(
        user_message=user_message,
        assistant_message=assistant_message,
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.siliconflow_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.siliconflow_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.siliconflow_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "stream": False,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Memory intent: JSON parse failed, raw=%s", raw[:200])
                return False, ""

            needs = parsed.get("needs_long_term_memory", False)
            content = parsed.get("memory_content", "")

            if not isinstance(needs, bool) or not isinstance(content, str):
                logger.warning("Memory intent: unexpected types needs=%s content=%s", type(needs), type(content))
                return False, ""

            if needs and content.strip():
                logger.info("Memory intent: WILL store — %s", content.strip()[:100])
                return True, content.strip()
            else:
                logger.debug("Memory intent: skip (needs=%s, content_empty=%s)", needs, not content.strip())
                return False, ""

    except httpx.HTTPStatusError as e:
        logger.warning("Memory intent: HTTP %s — %s", e.response.status_code, e.response.text[:200])
        return False, ""
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.warning("Memory intent: request failed — %s", e)
        return False, ""
    except Exception:
        logger.exception("Memory intent: unexpected error")
        return False, ""
