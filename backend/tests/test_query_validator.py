import pytest
from app.services.query_validator import validate_and_enhance_query, is_low_quality_result


class TestValidateAndEnhanceQuery:
    def test_single_character_query_gets_enhanced(self):
        result = validate_and_enhance_query("拉", "拉康的理论是什么")
        assert result.was_enhanced is True
        assert "拉康" in result.query

    def test_proper_noun_split_gets_fixed(self):
        result = validate_and_enhance_query("拉的理论", "拉康的理论是什么")
        assert "拉康" in result.query
        assert result.was_enhanced is True

    def test_valid_query_passes_through(self):
        # 贝多芬不在 DOMAIN_KEYWORDS 中，不会触发领域词追加
        result = validate_and_enhance_query("贝多芬 音乐", "贝多芬的音乐风格")
        assert result.was_enhanced is False
        assert result.query == "贝多芬 音乐"

    def test_short_query_gets_enhanced_from_user_message(self):
        result = validate_and_enhance_query("弗", "弗洛伊德潜意识理论")
        assert "弗洛伊德" in result.query
        # 从 user_message 提取专有名词 + 追加领域关键词 → 包含"精神分析"
        assert "精神分析" in result.query

    def test_enhancement_adds_domain_keywords(self):
        result = validate_and_enhance_query("拉康", "介绍一下拉康")
        assert "精神分析" in result.query or "镜像阶段" in result.query


class TestIsLowQualityResult:
    def test_dictionary_url_detected(self):
        assert is_low_quality_result({"url": "https://www.zdic.net/hans/拉", "title": "拉的字义"})

    def test_pinyin_url_detected(self):
        assert is_low_quality_result({"url": "https://example.com/pinyin/la", "title": "拼音"})

    def test_normal_result_passes(self):
        assert not is_low_quality_result({"url": "https://zh.wikipedia.org/wiki/拉康", "title": "拉康 - 维基百科"})
