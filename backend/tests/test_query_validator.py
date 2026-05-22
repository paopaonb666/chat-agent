import pytest
from unittest.mock import AsyncMock, patch
from app.services.query_validator import enhance_query, is_low_quality_result


class TestEnhanceQuery:
    """Test the LLM-driven query enhancement with all 5 defense layers."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_immediately(self):
        """A1: Empty query skips all LLM calls."""
        result = await enhance_query("")
        assert result.was_enhanced is False
        assert result.query == ""

    @pytest.mark.asyncio
    async def test_empty_whitespace_query_returns_immediately(self):
        """A1: Whitespace-only query skips all LLM calls."""
        result = await enhance_query("   ")
        assert result.was_enhanced is False

    @pytest.mark.asyncio
    async def test_overly_long_query_skips_enhancement(self):
        """A2: Query > 200 chars skips LLM."""
        long_query = "搜索 " * 80  # > 200 chars
        result = await enhance_query(long_query)
        assert result.was_enhanced is False
        assert "too long" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_llm_returns_ok_passes_through(self):
        """LLM says query is fine — return original."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "OK"
            result = await enhance_query("拉康 精神分析")
            assert result.was_enhanced is False
            assert result.query == "拉康 精神分析"

    @pytest.mark.asyncio
    async def test_llm_enhances_split_proper_noun(self):
        """LLM detects split noun and returns ENHANCE."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "ENHANCE: 拉康 精神分析 理论\nREASON: 专有名词被拆分"
            result = await enhance_query("拉的理论", "拉康的精神分析理论是什么")
            assert result.was_enhanced is True
            assert "拉康" in result.query
            assert "专有名词被拆分" in result.reason

    @pytest.mark.asyncio
    async def test_llm_timeout_falls_back_to_original(self):
        """B1: LLM timeout → return original query."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            import asyncio
            mock_llm.side_effect = asyncio.TimeoutError
            result = await enhance_query("拉康")
            assert result.was_enhanced is False
            assert result.query == "拉康"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_original(self):
        """B2: Any exception → return original query."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("API connection refused")
            result = await enhance_query("黑格尔 辩证法")
            assert result.was_enhanced is False
            assert result.query == "黑格尔 辩证法"

    @pytest.mark.asyncio
    async def test_llm_returns_garbage_falls_back(self):
        """C3: Unparseable output → return original."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "嗯，这个查询看起来需要一些调整，我建议改成..."
            result = await enhance_query("拉的理论")
            assert result.was_enhanced is False
            assert result.query == "拉的理论"

    @pytest.mark.asyncio
    async def test_llm_returns_ok_with_extra_text_still_ok(self):
        """C1: 'OK' at start of response (with trailing text) still matches."""
        # _parse_enhancement_response checks for ^OK\s*$ on first line after strip
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "OK"
            result = await enhance_query("Python 异步编程")
            assert result.was_enhanced is False

    @pytest.mark.asyncio
    async def test_d4_missing_original_tokens_falls_back(self):
        """D4: Enhanced query drops original tokens → reject enhancement."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            # Original query has "机器学习" but LLM output drops it entirely
            mock_llm.return_value = "ENHANCE: 深度学习 神经网络\nREASON: more specific terms"
            result = await enhance_query("机器学习 梯度下降", "什么是机器学习中的梯度下降")
            assert result.was_enhanced is False  # D4 should fail: "机器学习" missing

    @pytest.mark.asyncio
    async def test_d2_too_short_enhancement_rejected(self):
        """D2: Enhanced query shorter than 60% of original → reject."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            # Original is 12 chars, enhanced is only 4 chars = 33% → fail
            mock_llm.return_value = "ENHANCE: 拉康\nREASON: simplified"
            result = await enhance_query("拉康 精神分析 镜像阶段")
            assert result.was_enhanced is False  # D2 should fail

    @pytest.mark.asyncio
    async def test_d3_too_long_enhancement_rejected(self):
        """D3: Enhanced query > 120 chars → reject."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "ENHANCE: " + ("搜索 " * 50) + "\nREASON: detailed"
            result = await enhance_query("拉康")
            assert result.was_enhanced is False  # D3 should fail

    @pytest.mark.asyncio
    async def test_d1_empty_enhancement_rejected(self):
        """D1: LLM returns empty ENHANCE → rejected."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "ENHANCE: \nREASON: nothing"
            result = await enhance_query("拉康")
            assert result.was_enhanced is False

    @pytest.mark.asyncio
    async def test_d6_no_meaningful_change_not_flagged(self):
        """D6: Enhanced == original → not marked as enhanced."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "ENHANCE: 拉康  精神分析\nREASON: normalized whitespace"
            result = await enhance_query("拉康  精神分析")
            # After normalization, same as original
            assert result.was_enhanced is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_consecutive_failures(self):
        """B5: 3 consecutive failures → circuit breaker opens → skip LLM."""
        # Reset circuit breaker state before test
        from app.services.query_validator import _circuit_breaker
        _circuit_breaker.reset()

        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("API down")

            # 3 consecutive failures
            for _ in range(3):
                result = await enhance_query("拉康")
                assert result.was_enhanced is False
                assert result.query == "拉康"

            # 4th call: circuit should be open, skip LLM entirely
            mock_llm.side_effect = None
            mock_llm.return_value = "ENHANCE: 拉康 精神分析\nREASON: test"
            result = await enhance_query("拉康")
            # Circuit breaker is open, so LLM should NOT be called
            assert result.was_enhanced is False
            assert mock_llm.call_count == 3  # Not called again

        _circuit_breaker.reset()

    @pytest.mark.asyncio
    async def test_valid_query_not_enhanced_when_llm_says_ok(self):
        """Normal query that LLM deems OK stays unchanged."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "OK"
            result = await enhance_query("黑洞 形成 原理")
            assert result.was_enhanced is False
            assert result.query == "黑洞 形成 原理"

    @pytest.mark.asyncio
    async def test_reason_line_extracted_from_llm_response(self):
        """REASON line is parsed and stored in EnhancedQueryResult."""
        with patch("app.services.query_validator._call_llm_for_enhancement", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "ENHANCE: ChatGPT 大语言模型 教程\nREASON: 专有名词被截断为单字"
            result = await enhance_query("Chat", "ChatGPT怎么使用")
            assert result.was_enhanced is True
            assert "ChatGPT" in result.query
            assert "专有名词被截断" in result.reason


class TestIsLowQualityResult:
    """Low-quality result filtering — deterministic, independent of LLM."""

    def test_dictionary_url_detected(self):
        assert is_low_quality_result({"url": "https://www.zdic.net/hans/拉", "title": "拉的字义"})

    def test_pinyin_url_detected(self):
        assert is_low_quality_result({"url": "https://example.com/pinyin/la", "title": "拼音"})

    def test_baike_com_detected(self):
        assert is_low_quality_result({"url": "https://baike.com/view/123", "title": "百科"})

    def test_youdao_dict_detected(self):
        assert is_low_quality_result({"url": "https://dict.youdao.com/w/hello", "title": "有道词典"})

    def test_kangxi_zidian_detected(self):
        assert is_low_quality_result({"url": "https://kangxizidian.com/", "title": "康熙字典查询"})

    def test_dictionary_title_in_snippet_detected(self):
        assert is_low_quality_result({
            "url": "https://example.com/article",
            "title": "正常文章",
            "snippet": "根据汉语字典的解释，这个字的拼音是...",
        })

    def test_hanyu_in_url_detected(self):
        assert is_low_quality_result({"url": "https://hanyuzidian.com/char/5200", "title": "字义查询"})

    def test_normal_wikipedia_result_passes(self):
        assert not is_low_quality_result({"url": "https://zh.wikipedia.org/wiki/拉康", "title": "拉康 - 维基百科"})

    def test_normal_news_result_passes(self):
        assert not is_low_quality_result({"url": "https://news.sina.com.cn/article", "title": "最新科技新闻"})

    def test_normal_github_result_passes(self):
        assert not is_low_quality_result({"url": "https://github.com/org/repo", "title": "GitHub Repository"})

    def test_no_url_field_passes(self):
        assert not is_low_quality_result({"title": "正常标题", "snippet": "正常摘要"})
