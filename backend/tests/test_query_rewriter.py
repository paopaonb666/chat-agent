import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from app.services.query_rewriter import (
    RewriteStrategy,
    rewrite_query,
    _check_semantic_drift,
    _parse_rewritten_query,
    _format_history,
    CircuitBreaker,
)


class TestParseRewrittenQuery:
    def test_extract_from_rewritten_prefix(self):
        text = "REWRITTEN: 这是一个改写后的查询"
        result = _parse_rewritten_query(text)
        assert result == "这是一个改写后的查询"

    def test_extract_with_quotes(self):
        text = 'REWRITTEN: "quoted query"'
        result = _parse_rewritten_query(text)
        assert result == "quoted query"

    def test_single_line_no_prefix(self):
        text = "直接输出的查询"
        result = _parse_rewritten_query(text)
        assert result == "直接输出的查询"

    def test_empty_returns_none(self):
        assert _parse_rewritten_query("") is None
        assert _parse_rewritten_query("   ") is None


class TestCheckSemanticDrift:
    def test_no_drift_similar_query(self):
        is_drifted, reason = _check_semantic_drift(
            "拉康的精神分析理论", "关于拉康精神分析的主要观点"
        )
        assert is_drifted is False

    def test_drift_completely_different(self):
        is_drifted, reason = _check_semantic_drift(
            "深度学习框架", "古罗马建筑史"
        )
        assert is_drifted is True
        assert "drift" in reason.lower()

    def test_no_drift_short_query(self):
        is_drifted, _ = _check_semantic_drift("你好", "你好世界")
        assert is_drifted is False

    def test_empty_rewritten_is_drifted(self):
        is_drifted, reason = _check_semantic_drift("原始查询", "")
        assert is_drifted is True
        assert "empty" in reason.lower()


class TestFormatHistory:
    def test_empty_messages(self):
        result = _format_history([])
        assert result == "（无）"

    def test_single_turn(self):
        messages = [
            {"role": "user", "content": "什么是 RAG？"},
            {"role": "assistant", "content": "RAG 是检索增强生成。"},
        ]
        result = _format_history(messages)
        assert "什么是 RAG？" in result
        assert "RAG 是检索增强生成" in result

    def test_max_turns_limit(self):
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"问题{i}"})
            messages.append({"role": "assistant", "content": f"回答{i}"})

        result = _format_history(messages, max_turns=2)
        assert "问题8" in result
        assert "问题9" in result
        assert "问题0" not in result


class TestCircuitBreaker:
    def test_closed_by_default(self):
        cb = CircuitBreaker()
        assert cb.is_open is False

    def test_opens_after_failures(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        assert cb.is_open is False
        cb.record_failure()
        assert cb.is_open is True

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
        cb.record_failure()
        assert cb.is_open is True
        import time

        time.sleep(0.06)
        assert cb.is_open is False  # should become HALF_OPEN

    def test_success_resets(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.is_open is False


class TestRewriteQuery:
    @pytest.fixture
    def mock_settings(self):
        with patch(
            "app.services.query_rewriter.settings"
        ) as mock_s:
            mock_s.siliconflow_api_key = "fake-key"
            mock_s.siliconflow_base_url = "https://fake.api/v1"
            mock_s.siliconflow_model = "fake-model"
            yield mock_s

    @pytest.mark.asyncio
    async def test_empty_query_returns_unchanged(self):
        result = await rewrite_query("", strategy=RewriteStrategy.CONTEXT)
        assert result.query == ""
        assert result.was_rewritten is False

    @pytest.mark.asyncio
    async def test_long_query_returns_unchanged(self):
        long_query = "x" * 400
        result = await rewrite_query(long_query)
        assert result.query == long_query
        assert result.was_rewritten is False
        assert "too long" in result.reason

    @pytest.mark.asyncio
    async def test_no_api_key_returns_unchanged(self, mock_settings):
        mock_settings.siliconflow_api_key = ""
        result = await rewrite_query("测试查询")
        assert result.query == "测试查询"
        assert result.was_rewritten is False
        assert "no API key" in result.reason

    @pytest.mark.asyncio
    async def test_successful_rewrite(self, mock_settings):
        mock_response = "REWRITTEN: 这个Transformer模型到底是什么"

        with patch(
            "app.services.query_rewriter._call_llm_for_rewrite",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await rewrite_query(
                "Transformer是什么",
                messages=[
                    {"role": "user", "content": "介绍一下 Transformer"},
                    {"role": "assistant", "content": "Transformer 是一种神经网络架构。"},
                ],
                strategy=RewriteStrategy.CONTEXT,
            )
            assert result.was_rewritten is True
            assert result.query == "这个Transformer模型到底是什么"
            assert result.strategy == RewriteStrategy.CONTEXT

    @pytest.mark.asyncio
    async def test_drift_fallback(self, mock_settings):
        # LLM 返回完全无关的内容，应触发漂移检测回退
        mock_response = "REWRITTEN: 古罗马竞技场旅游攻略"

        with patch(
            "app.services.query_rewriter._call_llm_for_rewrite",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await rewrite_query(
                "深度学习的优化算法",
                strategy=RewriteStrategy.CONTEXT,
            )
            assert result.was_rewritten is False
            assert result.query == "深度学习的优化算法"
            assert "semantic drift" in result.reason

    @pytest.mark.asyncio
    async def test_timeout_fallback(self, mock_settings):
        with patch(
            "app.services.query_rewriter._call_llm_for_rewrite",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError,
        ):
            result = await rewrite_query("测试查询")
            assert result.was_rewritten is False
            assert result.query == "测试查询"
            assert "timeout" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks(self, mock_settings):
        cb = CircuitBreaker(failure_threshold=1)
        with patch(
            "app.services.query_rewriter._circuit_breaker", cb
        ):
            cb.record_failure()
            result = await rewrite_query("测试查询")
            assert result.was_rewritten is False
            assert "circuit breaker" in result.reason

    @pytest.mark.asyncio
    async def test_hyde_strategy(self, mock_settings):
        mock_response = "假设答案：Transformer 使用自注意力机制处理序列数据"

        with patch(
            "app.services.query_rewriter._call_llm_for_rewrite",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await rewrite_query(
                "Transformer 的原理",
                strategy=RewriteStrategy.HYDE,
            )
            assert result.was_rewritten is True
            assert result.strategy == RewriteStrategy.HYDE

    @pytest.mark.asyncio
    async def test_stepback_strategy(self, mock_settings):
        mock_response = "REWRITTEN: Transformer架构的自注意力机制是什么；Transformer架构相比RNN有什么优势"

        with patch(
            "app.services.query_rewriter._call_llm_for_rewrite",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await rewrite_query(
                "Transformer架构",
                strategy=RewriteStrategy.STEPBACK,
            )
            assert result.was_rewritten is True
            assert result.strategy == RewriteStrategy.STEPBACK
            assert "；" in result.query
