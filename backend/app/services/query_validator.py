"""Query 校验与增强模块。

防护 LLM 生成低质量搜索 query（如将专有名词拆成单字），
并通过关键词增强提升搜索召回率。
"""

from dataclasses import dataclass

# 常见专有名词词典 —— 用于检测拆字行为并补全
PROPER_NOUNS = {
    "拉康", "弗洛伊德", "荣格", "阿德勒", "萨特", "海德格尔",
    "维特根斯坦", "罗尔斯", "哈贝马斯", "福柯", "德勒兹", "齐泽克",
    "黑格尔", "尼采", "康德", "马克思", "恩格斯", "列宁",
    "索绪尔", "皮亚杰", "班杜拉", "马斯洛", "斯金纳",
    "ChatGPT", "OpenAI", "GPT-4", "GPT-3", "Claude",
    "苏轼", "李白", "杜甫", "王阳明", "朱熹",
    "老子", "庄子", "孔子", "孟子", "墨子",
    "硅谷", "华尔街", "纳斯达克", "美联储",
    "贝多芬", "莫扎特", "巴赫", "肖邦",
}

# 低质量结果模式 —— 字典、拼音、机械生成的 SEO 页面
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

# 领域关键词映射 —— 用于自动增强 query
DOMAIN_KEYWORDS = {
    "拉康": ["精神分析", "镜像阶段", "实在界"],
    "弗洛伊德": ["精神分析", "潜意识", "梦的解析"],
    "荣格": ["分析心理学", "集体无意识", "原型"],
    "ChatGPT": ["人工智能", "大语言模型", "OpenAI"],
    "黑格尔": ["辩证法", "绝对精神", "哲学"],
    "尼采": ["权力意志", "超人", "哲学"],
    "康德": ["批判哲学", "先验", "伦理学"],
}


@dataclass(frozen=True)
class EnhancedQueryResult:
    query: str
    was_enhanced: bool
    reason: str = ""
    warnings: list[str] | None = None

    def __post_init__(self):
        if self.warnings is None:
            object.__setattr__(self, "warnings", [])


def validate_and_enhance_query(query: str, user_message: str = "") -> EnhancedQueryResult:
    """校验并增强搜索 query。

    逻辑：
    1. 检测单字/极短 query → 从 user_message 中提取关键词补全
    2. 检测专有名词被拆分（如"拉"→"拉康"）→ 补全专有名词
    3. 为已知领域名词追加领域关键词
    """
    original = query.strip()
    if not original:
        return EnhancedQueryResult(query=original, was_enhanced=False, reason="空 query")

    enhanced = original
    reasons = []
    warnings = []

    # 1. 单字/极短拦截
    if len(original) <= 1:
        reasons.append(f"query 过短（'{original}'），触发增强")
        extracted = _extract_keywords(user_message)
        if extracted:
            enhanced = " ".join(extracted)
        else:
            enhanced = original
        warnings.append(f"原始 query '{original}' 过短，已用用户消息关键词增强")

    # 2. 专有名词拆分检测
    for noun in PROPER_NOUNS:
        if noun not in enhanced and len(noun) > 1:
            # 如果 query 中出现了名词的首字或部分字，且整个名词在 user_message 中
            if any(c in enhanced for c in noun) and noun in user_message:
                enhanced = enhanced.replace(noun[0], noun, 1) if noun[0] in enhanced else f"{noun} {enhanced}"
                reasons.append(f"检测到专有名词拆分，补全 '{noun}'")
                warnings.append(f"请勿将专有名词 '{noun}' 拆成单字搜索")
                break
            # 如果整个名词在 user_message 中但 query 只取了一个字
            if noun in user_message and len(original) <= 2 and original in noun:
                enhanced = f"{noun} {_extract_keywords(user_message, exclude={noun})[0] if _extract_keywords(user_message, exclude={noun}) else ''}"
                reasons.append(f"query '{original}' 疑似 '{noun}' 的拆分")
                warnings.append(f"检测到专有名词 '{noun}' 被拆分，已自动补全")
                break

    # 3. 如果专有名词已完整出现，追加领域关键词
    for noun, keywords in DOMAIN_KEYWORDS.items():
        if noun in enhanced and not any(k in enhanced for k in keywords):
            enhanced = f"{enhanced} {keywords[0]}"
            reasons.append(f"为 '{noun}' 追加领域关键词 '{keywords[0]}'")
            break

    # 4. 最终清理
    enhanced = _clean_query(enhanced)

    if enhanced != original:
        return EnhancedQueryResult(
            query=enhanced,
            was_enhanced=True,
            reason="; ".join(reasons) if reasons else "query 已增强",
            warnings=warnings,
        )

    return EnhancedQueryResult(query=original, was_enhanced=False)


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


def _extract_keywords(text: str, exclude: set[str] | None = None) -> list[str]:
    """从用户消息中提取候选关键词（简单实现）。"""
    exclude = exclude or set()
    # 优先提取完整专有名词
    found = []
    for noun in PROPER_NOUNS:
        if noun in text and noun not in exclude:
            found.append(noun)
    # 按长度降序，优先匹配长名词
    found.sort(key=len, reverse=True)
    # 去重（避免子串重复）
    filtered = []
    for w in found:
        if not any(w in x and w != x for x in filtered):
            filtered.append(w)
    return filtered[:3]


def _clean_query(query: str) -> str:
    """清理 query 中的多余空白。"""
    return " ".join(query.split())
