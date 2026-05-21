"""
循环代理 — 通过质量评估和自动修正迭代调用LLM。

用一个开放式循环替换固定轮数的LLM循环：
1. 调用LLM（流式）并支持工具调用
2. 通过简短的非流式LLM调用来评估响应质量
3. 如果不满意，附加修正指令并重试
4. 当质量通过、达到最大迭代次数或陷入循环时退出
"""

import json
import asyncio
import logging
from typing import AsyncGenerator

import httpx

from app.core.config import settings
from app.core.metrics import get_llm_counter
from app.core.sse import step_line
from app.services.web_search import web_search, format_web_context

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 50
MAX_SAME_FAILURE = 3
EVAL_TIMEOUT = 8.0
SEARCH_TIMEOUT = 8.0

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "搜索互联网获取最新信息。当需要了解实时新闻、当前事件、"
            "最新数据或不确定的事实性知识时调用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题，使用中文或英文，尽量精简准确",
                }
            },
            "required": ["query"],
        },
    },
}


def _accumulate_tool_calls(tool_calls_by_index: dict, delta_tool_calls: list[dict]):
    for tc in delta_tool_calls:
        idx = tc.get("index", 0)
        if idx not in tool_calls_by_index:
            tool_calls_by_index[idx] = {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        entry = tool_calls_by_index[idx]
        if tc.get("id"):
            entry["id"] = tc["id"]
        if tc.get("function", {}).get("name"):
            entry["function"]["name"] = tc["function"]["name"]
        if tc.get("function", {}).get("arguments"):
            entry["function"]["arguments"] += tc["function"]["arguments"]


class LoopAgent:
    """带质量评估和自动修正的迭代式LLM代理。"""

    def __init__(self, max_iterations: int = MAX_ITERATIONS):
        self.max_iterations = max_iterations
        self.final_content = ""
        self._last_failure_reason = ""
        self._same_failure_count = 0

    async def run(
        self,
        llm_messages: list[dict],
        user_message: str,
        enable_web_search: bool,
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> AsyncGenerator[str, None]:
        tools = [WEB_SEARCH_TOOL] if enable_web_search else None

        for iteration in range(1, self.max_iterations + 1):
            tool_calls_by_index: dict[int, dict] = {}
            finish_reason = None
            assistant_content = ""

            # ── LLM call (streaming) ─────────────────────────────────
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model_name,
                            "messages": llm_messages,
                            "stream": True,
                            "tools": tools,
                        },
                        timeout=60.0,
                    ) as response:
                        if response.status_code != 200:
                            get_llm_counter().labels(model=model_name, status=str(response.status_code)).inc()
                            _ = await response.aread()
                            yield f"data: {json.dumps({'error': f'API returned {response.status_code}'})}\n\n"
                            return

                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                            except Exception:
                                logger.warning("LLM stream JSON parse failed")
                                continue

                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")

                            if "tool_calls" in delta:
                                _accumulate_tool_calls(tool_calls_by_index, delta["tool_calls"])

                            content = delta.get("content", "")
                            if content:
                                assistant_content += content
                                # Filter tool-call syntax from user-visible content
                                if not _has_tool_syntax(content):
                                    yield f"data: {json.dumps({'content': content})}\n\n"

                            if finish_reason in ("stop", "length", "content_filter"):
                                yield "data: [DONE]\n\n"
                                break

                get_llm_counter().labels(model=model_name, status="200").inc()

            except httpx.HTTPError:
                logger.exception("LLM HTTP error at iteration %d", iteration)
                yield step_line("loop_agent", "error", "Loop Agent", f"迭代 {iteration} 网络错误")
                continue

            # ── Handle tool calls ────────────────────────────────────
            any_web_sources = []
            if finish_reason == "tool_calls" and tool_calls_by_index:
                tc_list = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
                assistant_tc_msg = {
                    "role": "assistant",
                    "content": assistant_content or None,
                    "tool_calls": tc_list,
                }
                llm_messages.append(assistant_tc_msg)

                for tc in tc_list:
                    func_name = tc["function"]["name"]
                    if func_name == "web_search":
                        search_query = _resolve_search_query(tc, user_message)

                        yield step_line("web_search", "running", "联网搜索",
                                        f"Agent 请求搜索：{search_query[:40]}...")

                        try:
                            sources = await asyncio.wait_for(
                                web_search(search_query), timeout=SEARCH_TIMEOUT)
                        except Exception:
                            logger.exception("Web search failed")
                            sources = []

                        yield step_line("web_search", "completed", "联网搜索",
                                        f"找到 {len(sources)} 条结果")
                        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

                        any_web_sources.extend(sources)

                        tool_result = format_web_context(sources) or "未找到相关搜索结果。"
                        llm_messages.append({
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tc["id"],
                        })

                # Strip tools so second LLM call in same iteration is text-only
                tools = None
                # Continue to next LLM call within this same iteration
                continue

            # ── No tool calls — evaluate & decide ────────────────────
            self.final_content = assistant_content

            if not assistant_content.strip():
                yield step_line("loop_agent", "error", "Loop Agent",
                                f"迭代 {iteration}: LLM 返回空内容")
                break

            # Evaluate quality
            is_ok, reason = await _evaluate_quality(
                user_message, assistant_content, api_key, base_url, model_name)

            yield step_line(
                "loop_agent",
                "completed" if is_ok else "running",
                "Loop Agent",
                f"迭代 {iteration}/{self.max_iterations}: {'通过' if is_ok else '未通过 — ' + reason[:50]}"
            )

            if is_ok:
                return  # Success

            # Track repeated failures
            if reason == self._last_failure_reason:
                self._same_failure_count += 1
            else:
                self._same_failure_count = 1
                self._last_failure_reason = reason

            if self._same_failure_count >= MAX_SAME_FAILURE:
                yield step_line("loop_agent", "error", "Loop Agent",
                                f"连续 {MAX_SAME_FAILURE} 次相同失败，停止迭代")
                break

            # Append correction
            correction = _build_correction(reason)
            llm_messages.append({"role": "system", "content": correction})
            tools = [WEB_SEARCH_TOOL] if enable_web_search else None

        # Max iterations exhausted
        yield step_line("loop_agent", "error", "Loop Agent",
                        f"达到最大迭代次数 {self.max_iterations}，请用户介入")


# ── 内部辅助函数 ────────────────────────────────────────────────────────

def _resolve_search_query(tc: dict, fallback_message: str) -> str:
    try:
        args = json.loads(tc["function"]["arguments"])
        query = args.get("query", "")
        if not query or len(query.strip()) <= 3:
            return fallback_message
        return query
    except Exception:
        return fallback_message


async def _evaluate_quality(
    user_message: str,
    assistant_response: str,
    api_key: str,
    base_url: str,
    model_name: str,
) -> tuple[bool, str]:
    """快速非流式评估：响应是否回答了用户的问题？"""
    prompt = (
        f'用户问题是："{user_message}"\n\n'
        f'以下是助手的回答：\n{assistant_response[:800]}\n\n'
        f'请判断助手的回答是否直接、正确地回应了用户的问题。'
        f'如果回答与用户问题无关（例如用户问哲学理论却返回了汉字字典解释），请回复 FAIL: <简短原因>。'
        f'如果回答基本正确回应了用户问题，请回复 OK。'
        f'只回复 OK 或 FAIL: <原因>，不要回复其他内容。'
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 80,
                },
                timeout=EVAL_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning("Evaluate API returned %d", resp.status_code)
                return True, ""  # Be lenient on API error

            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            logger.debug("Evaluate result: %s", text)

            if text.upper().startswith("OK"):
                return True, ""
            # Extract reason after FAIL:
            reason = text.split("FAIL:", 1)[-1].strip() if "FAIL:" in text else text[:80]
            return False, reason

    except Exception:
        logger.exception("Evaluation call failed")
        return True, ""  # Be lenient — don't block on eval errors


def _build_correction(reason: str) -> str:
    return (
        f"上一轮你的回答被判定为不合格。原因：{reason}\n\n"
        f"请修正后重新回答。特别注意：\n"
        f"1. 搜索时必须使用完整的人名、地名或专有名词，禁止拆成单字\n"
        f"2. 确保回答内容与用户问题直接相关\n"
        f"3. 如果之前的搜索词不够准确，请用更完整的关键词重新调用 web_search"
    )


# ── 内容过滤器 ──────────────────────────────────────────────────────────

import re

_TOOL_SYNTAX_RE = re.compile(
    r'<(function_calls|invoke|tool_call|parameter|xml|dsml)>|'
    r'</(function_calls|invoke|tool_call|parameter|xml|dsml)>|'
    r'```(json|xml|dsml|tool_calls)\s*\n.*?\n```|'
    r'\bweb_search\s*\(\s*\{.*?\}\s*\)|'
    r'<\w+>\s*<parameter[^>]*>',
    re.DOTALL,
)


def _has_tool_syntax(text: str) -> bool:
    return bool(_TOOL_SYNTAX_RE.search(text))
