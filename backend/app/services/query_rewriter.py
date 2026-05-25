"""Query 改写模块 — 基于 LLM 的多策略查询重写。

支持三种改写策略：
1. HYDE：生成假设性答案作为检索 query
2. STEPBACK：将复杂问题分解为子问题
3. CONTEXT：整合对话历史，消解指代和省略

架构（5 层防御）：
  A. 前置校验（确定性规则）
  B. 外部调用（超时 + 异常隔离 + 熔断器）
  C. 输出解析（格式提取）
  D. 后置验证（语义漂移检测）
  E. 可观测性（分层日志 + 指标）

核心原则：任一层失败 → 返回原始 query，绝不阻塞 RAG 流程（fail-open）。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
REWRITE_TIMEOUT = 10.0
MAX_REWRITE_LENGTH = 300
SEMANTIC_DRIFT_THRESHOLD = 0.5
MAX_HISTORY_TURNS = 3


# ---------------------------------------------------------------------------
# 熔断器
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """简单的熔断器，模块级单例。

    状态机: CLOSED → OPEN (3 次连续失败) → HALF_OPEN (60s 后) → CLOSED / OPEN
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        if self._state == "CLOSED":
            return False
        if self._state == "OPEN":
            if time.monotonic() - self._opened_at >= self._cooldown_seconds:
                self._state = "HALF_OPEN"
                logger.info("QueryRewriter CB: OPEN → HALF_OPEN")
                return False
            return True
        # HALF_OPEN — allow one probe request
        return False

    def record_success(self) -> None:
        if self._state == "HALF_OPEN":
            self._state = "CLOSED"
            self._failure_count = 0
            logger.info("QueryRewriter CB: HALF_OPEN → CLOSED")
        elif self._state == "CLOSED":
            self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._state == "HALF_OPEN":
            self._state = "OPEN"
            self._opened_at = time.monotonic()
            logger.error("QueryRewriter CB: HALF_OPEN → OPEN")
        elif self._state == "CLOSED" and self._failure_count >= self._failure_threshold:
            self._state = "OPEN"
            self._opened_at = time.monotonic()
            logger.error(
                "QueryRewriter CB: CLOSED → OPEN (%d failures, cool %ds)",
                self._failure_count,
                self._cooldown_seconds,
            )


_circuit_breaker = CircuitBreaker()


# ---------------------------------------------------------------------------
# 数据结构与枚举
# ---------------------------------------------------------------------------
class RewriteStrategy(Enum):
    """Query 改写策略枚举。"""

    HYDE = "hyde"
    """假设答案生成：让 LLM 先生成一个假设答案，再用该答案作为检索 query。"""

    STEPBACK = "stepback"
    """任务分解：将复杂问题拆分为 2-3 个更基础的子问题。"""

    CONTEXT = "context"
    """上下文补全：整合对话历史，消解指代和省略。"""


@dataclass
class RewriteResult:
    """Query 改写结果。"""

    query: str
    was_rewritten: bool = False
    strategy: RewriteStrategy | None = None
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
_HYDE_PROMPT = """你是一个智能检索助手。你的任务是根据用户的问题，生成一个假设性的直接答案。这个假设答案将用于在历史对话中检索相关信息。

要求：
- 答案应尽可能具体、包含关键术语和实体名称
- 答案长度控制在 50-150 字
- 不要反问，直接给出假设性答案
- 如果问题涉及专业概念，答案中应包含这些概念的全称

对话历史：
{history}

当前问题：{query}

请直接输出假设答案，不要添加解释。"""

_STEPBACK_PROMPT = """你是一个问题分解专家。你的任务是将用户的复杂问题拆分为 2-3 个更基础、更具体的子问题，以便分别检索相关信息。

要求：
- 每个子问题必须独立完整，不依赖上下文中的省略
- 子问题之间用中文分号（；）分隔
- 保留原问题中的核心实体（人名、地名、专有名词）
- 不要添加原问题中没有的信息

对话历史：
{history}

当前问题：{query}

请直接输出子问题列表，格式为：
REWRITTEN: 子问题1；子问题2；子问题3"""

_CONTEXT_PROMPT = """你是一个对话理解助手。你的任务是根据对话历史，将用户的当前问题补全为一个完整、独立的查询语句，消除所有指代和省略。

需要处理的场景：
- 指代消解：将"它""那个""他""她"等替换为具体实体
- 省略补全：补充被省略的主语、宾语或限定条件
- 上下文继承：将依赖前文语境的隐含条件显式写出

要求：
- 补全后的查询必须独立可理解，不依赖对话历史
- 保留用户原始意图，不改变核心含义
- 输出格式严格为：REWRITTEN: <补全后的查询>

对话历史：
{history}

当前问题：{query}

请直接输出补全后的查询。"""


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def _parse_rewritten_query(text: str) -> str | None:
    """C 层：从 LLM 响应中提取改写后的 query。

    Returns:
        提取到的 query 字符串，若解析失败返回 None。
    """
    clean = text.strip()
    if not clean:
        return None

    # 优先匹配 REWRITTEN: 前缀
    if "REWRITTEN:" in clean:
        parts = clean.split("REWRITTEN:", 1)
        if len(parts) == 2:
            candidate = parts[1].strip()
            # 去除可能的引号
            if candidate.startswith('"') and candidate.endswith('"'):
                candidate = candidate[1:-1]
            if candidate.startswith("'") and candidate.endswith("'"):
                candidate = candidate[1:-1]
            if candidate:
                return candidate

    # 如果没有 REWRITTEN: 前缀，但整体是一行有效文本，也接受
    # （适用于 HYDE 策略直接输出答案的情况）
    if "\n" not in clean and len(clean) > 3:
        return clean

    # 取最后一行非空内容作为 fallback
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return None


# ---------------------------------------------------------------------------
# 语义漂移检测（D 层）
# ---------------------------------------------------------------------------
def _check_semantic_drift(original: str, rewritten: str) -> tuple[bool, str]:
    """检测改写后的 query 是否发生了语义漂移。

    检测逻辑：
    1. 分词后计算核心词（长度>1）的保留率
    2. 若保留率低于阈值，判定为漂移

    Args:
        original: 原始 query
        rewritten: 改写后的 query

    Returns:
        (is_drifted, reason) — is_drifted=True 表示发生漂移，应回退到原始 query
    """
    original = original.strip()
    rewritten = rewritten.strip()

    if not original or not rewritten:
        return True, "empty query after rewrite"

    # 尝试 jieba 分词，失败则按字符/空格处理
    try:
        import jieba

        orig_tokens = [t for t in jieba.lcut(original) if len(t) > 1]
        rew_tokens = [t for t in jieba.lcut(rewritten) if len(t) > 1]
    except Exception:
        # 空格分词 fallback
        orig_tokens = [t for t in original.split() if len(t) > 1]
        rew_tokens = [t for t in rewritten.split() if len(t) > 1]

    if not orig_tokens:
        # 原始 query 太短，无法判定，保守地不认为漂移
        return False, ""

    # 计算核心词保留率
    preserved = sum(1 for t in orig_tokens if t in rew_tokens)
    retention_rate = preserved / len(orig_tokens)

    if retention_rate < SEMANTIC_DRIFT_THRESHOLD:
        return (
            True,
            f"semantic drift detected (retention={retention_rate:.2f} < {SEMANTIC_DRIFT_THRESHOLD})",
        )

    return False, ""


# ---------------------------------------------------------------------------
# LLM 调用（B 层）
# ---------------------------------------------------------------------------
async def _call_llm_for_rewrite(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    """调用 SiliconFlow LLM 获取改写结果。"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 512,
                "stream": False,
            },
            timeout=REWRITE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 历史格式化
# ---------------------------------------------------------------------------
def _format_history(messages: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """将消息列表格式化为对话历史文本。"""
    if not messages:
        return "（无）"

    # 取最近 max_turns 轮（每轮包含 user + assistant）
    # 先找出所有 user 消息的索引，再取最后 max_turns 个 user 消息及其后的 assistant
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    selected_indices = user_indices[-max_turns:] if len(user_indices) > max_turns else user_indices

    lines = []
    for idx in selected_indices:
        user_msg = messages[idx].get("content", "")
        lines.append(f"用户：{user_msg}")
        # 尝试获取同一轮的 assistant 回复
        if idx + 1 < len(messages) and messages[idx + 1].get("role") == "assistant":
            assistant_msg = messages[idx + 1].get("content", "")
            lines.append(f"助手：{assistant_msg}")

    return "\n".join(lines) if lines else "（无）"


# ---------------------------------------------------------------------------
# 各策略实现
# ---------------------------------------------------------------------------
async def _hyde_rewrite(
    query: str,
    messages: list[dict],
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[str | None, str]:
    """HyDE 改写：生成假设答案作为检索 query。"""
    history = _format_history(messages)
    prompt = _HYDE_PROMPT.format(query=query, history=history)

    response = await _call_llm_for_rewrite(prompt, api_key, base_url, model)
    rewritten = _parse_rewritten_query(response)

    if rewritten is None:
        return None, "hyde parse failed"

    # HyDE 生成的是答案，可能较长，截断到合理长度
    if len(rewritten) > MAX_REWRITE_LENGTH:
        rewritten = rewritten[:MAX_REWRITE_LENGTH]

    return rewritten, "hyde generated"


async def _stepback_rewrite(
    query: str,
    messages: list[dict],
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[str | None, str]:
    """Stepback 改写：将复杂问题拆分为子问题。"""
    history = _format_history(messages)
    prompt = _STEPBACK_PROMPT.format(query=query, history=history)

    response = await _call_llm_for_rewrite(prompt, api_key, base_url, model)
    rewritten = _parse_rewritten_query(response)

    if rewritten is None:
        return None, "stepback parse failed"

    # 验证是否包含分号（子问题分隔符）
    if "；" not in rewritten and ";" not in rewritten:
        logger.warning("Stepback: no semicolon in output, treating as single query")

    return rewritten, "stepback decomposed"


async def _context_rewrite(
    query: str,
    messages: list[dict],
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[str | None, str]:
    """Context 改写：整合对话历史，消解指代。"""
    history = _format_history(messages)
    prompt = _CONTEXT_PROMPT.format(query=query, history=history)

    response = await _call_llm_for_rewrite(prompt, api_key, base_url, model)
    rewritten = _parse_rewritten_query(response)

    if rewritten is None:
        return None, "context parse failed"

    return rewritten, "context resolved"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
async def rewrite_query(
    query: str,
    messages: list[dict] | None = None,
    strategy: RewriteStrategy = RewriteStrategy.CONTEXT,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> RewriteResult:
    """改写用户 query 以提升 RAG 检索效果。

    Args:
        query: 用户原始查询
        messages: 对话历史，格式为 [{"role": "user"/"assistant", "content": "..."}, ...]
        strategy: 改写策略，默认 CONTEXT
        api_key: LLM API key（可选，默认从 settings 读取）
        base_url: LLM API base URL（可选）
        model: LLM model name（可选）

    Returns:
        RewriteResult — was_rewritten=True 表示改写成功并通过漂移检测。
        任何层失败时返回原始 query + was_rewritten=False（fail-open）。
    """
    # ── A 层：前置校验 ───────────────────────────────────
    original = query.strip()
    if not original:
        return RewriteResult(query=original, was_rewritten=False, reason="empty query")

    if len(original) > 300:
        return RewriteResult(query=original, was_rewritten=False, reason="query too long, skip")

    # ── B5 熔断器检查 ───────────────────────────────────
    if _circuit_breaker.is_open:
        return RewriteResult(
            query=original,
            was_rewritten=False,
            reason="circuit breaker open, skipping rewrite",
        )

    # ── B 层：LLM 调用 ───────────────────────────────────
    key = api_key or settings.siliconflow_api_key
    url = base_url or settings.siliconflow_base_url
    m = model or settings.siliconflow_model
    if not key:
        logger.warning("Query rewrite: no API key configured, skipping")
        return RewriteResult(query=original, was_rewritten=False, reason="no API key")

    messages = messages or []

    try:
        if strategy == RewriteStrategy.HYDE:
            rewritten, detail = await asyncio.wait_for(
                _hyde_rewrite(original, messages, key, url, m),
                timeout=REWRITE_TIMEOUT,
            )
        elif strategy == RewriteStrategy.STEPBACK:
            rewritten, detail = await asyncio.wait_for(
                _stepback_rewrite(original, messages, key, url, m),
                timeout=REWRITE_TIMEOUT,
            )
        else:  # CONTEXT
            rewritten, detail = await asyncio.wait_for(
                _context_rewrite(original, messages, key, url, m),
                timeout=REWRITE_TIMEOUT,
            )
    except asyncio.TimeoutError:
        logger.warning("Query rewrite: LLM call timed out after %.1fs", REWRITE_TIMEOUT)
        _circuit_breaker.record_failure()
        return RewriteResult(query=original, was_rewritten=False, reason="LLM timeout")
    except Exception as exc:
        logger.warning("Query rewrite: LLM call failed — %s: %s", type(exc).__name__, exc)
        _circuit_breaker.record_failure()
        return RewriteResult(query=original, was_rewritten=False, reason=f"LLM error: {exc}")

    if rewritten is None:
        _circuit_breaker.record_success()  # LLM 返回了，但解析失败
        return RewriteResult(
            query=original,
            was_rewritten=False,
            reason="parse failed after LLM call",
        )

    # ── D 层：语义漂移检测 ───────────────────────────────
    is_drifted, drift_reason = _check_semantic_drift(original, rewritten)
    if is_drifted:
        _circuit_breaker.record_success()
        logger.warning(
            "Query rewrite: drift detected, fallback to original — %s",
            drift_reason,
        )
        return RewriteResult(
            query=original,
            was_rewritten=False,
            reason=f"semantic drift: {drift_reason}",
            warnings=[drift_reason],
        )

    # ── E 层：成功日志 ───────────────────────────────────
    _circuit_breaker.record_success()
    logger.info(
        "Query rewritten: '%s' → '%s' via %s (%s)",
        original,
        rewritten,
        strategy.value,
        detail,
    )

    return RewriteResult(
        query=rewritten,
        was_rewritten=True,
        strategy=strategy,
        reason=detail,
    )
