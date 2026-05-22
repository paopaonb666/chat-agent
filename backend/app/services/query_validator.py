"""Query 校验与增强模块 — LLM 驱动 + 5 层防御。

防护 LLM 生成低质量搜索 query（如将专有名词拆成单字），
并通过 LLM 动态增强提升搜索召回率。

架构：
  A. 前置校验 (确定性规则)
  B. LLM 调用 (超时 + 异常隔离 + 熔断)
  C. 输出解析 (多格式兼容 + 正则校验)
  D. 后置验证 (6 条硬规则)
  E. 可观测性 (分层日志)

核心原则：任一层失败 → 返回原始 query，绝不阻塞搜索 (fail-open)。
"""

import re
import time
import asyncio
import logging
from dataclasses import dataclass, field

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ENHANCE_TIMEOUT = 5.0
MAX_ENHANCED_LENGTH = 120

# 低质量结果模式 —— 字典、拼音、机械生成的 SEO 页面（确定性规则，保留）
LOW_QUALITY_URL_PATTERNS = [
    "zdic.net", "chazidian", "cidian", "hanyu", "pinyin",
    "dxsbb.com", "baike.com", "youdao.com", "iciba.com",
    "hanyuzidian.com", "kangxizidian", "zidian",
]

LOW_QUALITY_TITLE_PATTERNS = [
    "汉语字典", "康熙字典", "拼音", "字义", "笔画", "部首",
    "五笔", "仓颉", "四角号码", "说文解字", "繁体字",
    "异体字", "同音字", "汉语词典", "英汉词典",
]

# ---------------------------------------------------------------------------
# 熔断器 (B5)
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
                logger.info("Circuit breaker: OPEN → HALF_OPEN (cooldown expired)")
                return False
            return True
        # HALF_OPEN — allow one probe request
        return False

    def record_success(self) -> None:
        if self._state == "HALF_OPEN":
            self._state = "CLOSED"
            self._failure_count = 0
            logger.info("Circuit breaker: HALF_OPEN → CLOSED (probe succeeded)")
        elif self._state == "CLOSED":
            self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._state == "HALF_OPEN":
            self._state = "OPEN"
            self._opened_at = time.monotonic()
            logger.error("Circuit breaker: HALF_OPEN → OPEN (probe failed)")
        elif self._state == "CLOSED" and self._failure_count >= self._failure_threshold:
            self._state = "OPEN"
            self._opened_at = time.monotonic()
            logger.error(
                "Circuit breaker: CLOSED → OPEN (%d consecutive failures, cooling for %ds)",
                self._failure_count,
                self._cooldown_seconds,
            )

    def reset(self) -> None:
        self._failure_count = 0
        self._state = "CLOSED"
        self._opened_at = 0.0


_circuit_breaker = CircuitBreaker()

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class EnhancedQueryResult:
    query: str
    was_enhanced: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM Enhancement Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a search query optimizer. Your ONLY task: check if a query needs to be rewritten.

A query needs ENHANCEMENT if:
1. A proper noun (person/place/technical term) is split into single characters.
   Example: "拉的理论" → should be "拉康 精神分析 理论"
2. The query is too short or too vague to return good results.
   Example: "Chat" → should be "ChatGPT" (if context says so)

A query is OK if it already contains complete, specific keywords.

RULES:
- NEVER add information not explicitly present in the user's message or query.
- NEVER change the core meaning.
- If query is already good, respond ONLY with "OK".
- If query needs enhancement, respond ONLY with:
  ENHANCE: <improved query>
  REASON: <one sentence>
- Keep enhanced query under 100 characters."""

_USER_PROMPT_TEMPLATE = "Query: {query}\nUser Message: {user_message}"

# ---------------------------------------------------------------------------
# 解析与验证 (C + D 层)
# ---------------------------------------------------------------------------
_RE_OK = re.compile(r"^OK\s*$")
_RE_GIBBERISH = re.compile(r"([^\x00-\x7f一-鿿㐀-䶿\s\w]{3,})")


def _parse_enhancement_response(text: str) -> tuple[str | None, str | None]:
    """C 层：解析 LLM 响应（逐行解析，避免正则贪婪/换行陷阱）。

    Returns:
        (enhanced_query, reason) — 均为 None 表示解析失败或 LLM 判断不需增强。
    """
    clean = text.strip()
    if not clean:
        return None, None

    first_line = clean.split("\n")[0].strip()
    if _RE_OK.match(first_line):
        return None, None

    enhanced: str | None = None
    reason: str | None = None

    for line in clean.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("ENHANCE:"):
            enhanced = stripped[len("ENHANCE:"):].strip()
        elif stripped.upper().startswith("REASON:"):
            reason = stripped[len("REASON:"):].strip()

    if enhanced is None:
        logger.warning("Query enhancement: unparseable LLM response: %s", clean[:120])
        return None, None

    return enhanced, reason or ""


def _validate_enhanced_query(original: str, enhanced: str) -> bool:
    """D 层：6 条硬规则验证。

    Returns:
        True 如果增强结果通过所有检查。
    """
    original = original.strip()
    enhanced = enhanced.strip()

    # D1 非空
    if len(enhanced) == 0:
        logger.warning("Query enhancement: D1 failed — enhanced query is empty")
        return False

    # D2 长度不缩水（允许最多缩 40%）
    if len(enhanced) < len(original) * 0.6:
        logger.warning(
            "Query enhancement: D2 failed — enhanced too short (orig=%d, enhanced=%d)",
            len(original), len(enhanced),
        )
        return False

    # D3 上限
    if len(enhanced) > MAX_ENHANCED_LENGTH:
        logger.warning(
            "Query enhancement: D3 failed — enhanced too long (%d > %d)",
            len(enhanced), MAX_ENHANCED_LENGTH,
        )
        return False

    # D4 原词保留
    try:
        import jieba
        tokens = [t for t in jieba.lcut(original) if len(t) > 1]
    except Exception:
        tokens = [t for t in original.split() if len(t) > 1]

    missing = [t for t in tokens if t not in enhanced]
    if len(missing) > 1:  # 允许 1 个词缺失（无意义虚词等）
        logger.warning(
            "Query enhancement: D4 failed — missing tokens: %s", missing,
        )
        return False

    # D5 无乱码
    if _RE_GIBBERISH.search(enhanced):
        logger.warning("Query enhancement: D5 failed — gibberish detected in '%s'", enhanced)
        return False

    # D6 有意义变化（忽略纯空格差异）
    if " ".join(enhanced.split()) == " ".join(original.split()):
        logger.info("Query enhancement: D6 — no meaningful change from original")
        return False

    return True


# ---------------------------------------------------------------------------
# LLM 调用 (B 层)
# ---------------------------------------------------------------------------
async def _call_llm_for_enhancement(
    query: str,
    user_message: str,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    """B 层：调用 LLM 获取增强建议，包含超时和异常隔离。"""
    system_prompt = _SYSTEM_PROMPT
    user_prompt = _USER_PROMPT_TEMPLATE.format(query=query, user_message=user_message)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 200,
                "stream": False,
            },
            timeout=ENHANCE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _resolve_llm_config(api_key: str, base_url: str, model: str) -> tuple[str, str, str]:
    """解析 LLM 凭据，未提供时从 settings 回退。"""
    key = api_key or settings.siliconflow_api_key
    url = base_url or settings.siliconflow_base_url
    m = model or settings.siliconflow_model
    return key, url, m


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
async def enhance_query(
    query: str,
    user_message: str = "",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> EnhancedQueryResult:
    """校验并增强搜索 query（LLM 驱动 + 5 层防御）。

    Args:
        query: LLM 生成的搜索 query（可能质量不佳）
        user_message: 用户原始消息，作为增强的上下文参考
        api_key: LLM API key（可选，默认从 settings 读取）
        base_url: LLM API base URL（可选）
        model: LLM model name（可选）

    Returns:
        EnhancedQueryResult — was_enhanced=True 表示 query 被成功增强。
        任何层失败时返回原始 query + was_enhanced=False。
    """
    # ── A 层：前置校验 ───────────────────────────────────
    original = query.strip()
    if not original:
        return EnhancedQueryResult(query=original, was_enhanced=False, reason="empty query")

    if len(original) > 200:
        return EnhancedQueryResult(query=original, was_enhanced=False, reason="query too long, skip")

    # ── B5 熔断器检查 ───────────────────────────────────
    if _circuit_breaker.is_open:
        return EnhancedQueryResult(
            query=original,
            was_enhanced=False,
            reason="circuit breaker open, skipping enhancement",
        )

    # ── B 层：LLM 调用 ───────────────────────────────────
    key, url, m = _resolve_llm_config(api_key, base_url, model)
    if not key:
        logger.warning("Query enhancement: no API key configured, skipping")
        return EnhancedQueryResult(query=original, was_enhanced=False, reason="no API key")

    try:
        llm_response = await asyncio.wait_for(
            _call_llm_for_enhancement(original, user_message, key, url, m),
            timeout=ENHANCE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Query enhancement: LLM call timed out after %.1fs", ENHANCE_TIMEOUT)
        _circuit_breaker.record_failure()
        return EnhancedQueryResult(query=original, was_enhanced=False, reason="LLM timeout")
    except Exception as exc:
        logger.warning("Query enhancement: LLM call failed — %s: %s", type(exc).__name__, exc)
        _circuit_breaker.record_failure()
        return EnhancedQueryResult(query=original, was_enhanced=False, reason=f"LLM error: {exc}")

    # ── C 层：输出解析 ───────────────────────────────────
    enhanced, reason = _parse_enhancement_response(llm_response)
    if enhanced is None:
        _circuit_breaker.record_success()  # LLM 返回 OK 也算成功
        return EnhancedQueryResult(query=original, was_enhanced=False, reason="LLM deemed query OK")

    # ── D 层：后置验证 ───────────────────────────────────
    if not _validate_enhanced_query(original, enhanced):
        _circuit_breaker.record_success()  # LLM 有输出，但被规则拦截
        return EnhancedQueryResult(
            query=original,
            was_enhanced=False,
            reason="post-validation failed, falling back to original",
        )

    # ── E 层：成功日志 ───────────────────────────────────
    _circuit_breaker.record_success()
    logger.info("Query enhanced: '%s' → '%s' (%s)", original, enhanced, reason or "enhanced")

    warnings: list[str] = []
    return EnhancedQueryResult(
        query=enhanced,
        was_enhanced=True,
        reason=reason or "query enhanced",
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 低质量结果过滤（确定性规则，保留）
# ---------------------------------------------------------------------------
def is_low_quality_result(result: dict) -> bool:
    """判断搜索结果是否为低质量页面（字典/拼音/SEO 垃圾页）。"""
    url = result.get("url", "").lower()
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()

    for pat in LOW_QUALITY_URL_PATTERNS:
        if pat in url:
            return True
    for pat in LOW_QUALITY_TITLE_PATTERNS:
        if pat in title or pat in snippet:
            return True
    return False
