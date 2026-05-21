"""
诊断脚本：直接测试记忆意图识别模型的 JSON 输出。
验证 SiliconFlow Qwen2.5-7B 是否正确输出 needs_long_term_memory 和 memory_content。

用法：
    cd backend && venv/Scripts/python scripts/diagnose_intent.py
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import httpx

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Same prompt as intent.py
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

test_cases = [
    # ── 应记忆的正样本 ──
    ("我叫炮炮", "很高兴认识你！", "自我介绍-姓名"),
    ("我住在北京朝阳区", "好的，记住了。", "住址信息"),
    ("我是一个程序员，在腾讯工作", "了解了你的职业。", "职业+公司"),
    ("我喜欢吃辣的，尤其是川菜", "川菜确实很美味！", "饮食偏好"),
    ("我对青霉素过敏", "我会记住这一点。", "健康-过敏"),
    ("我老婆叫小红，我们有两个孩子", "你的家庭很幸福！", "家庭关系"),
    ("我的职业目标是三年内成为架构师", "很有志向！", "长期目标"),
    ("我每天早上6点起床跑步", "好习惯！", "作息+运动习惯"),
    # ── 不应记忆的负样本 ──
    ("帮我搜索一下北京天气", "北京今天晴天25度。", "操作指令-搜索"),
    ("打开文件管理器", "文件管理器已打开。", "操作指令-打开"),
    ("你好", "你好，有什么可以帮助你？", "问候"),
    ("谢谢你的帮助", "不客气！", "致谢"),
    ("帮我把这段中文翻译成英文", "Here is the translation...", "翻译任务"),
    ("帮我写一封辞职信", "好的，这是模板...", "写文档任务"),
    ("什么是量子计算", "量子计算是利用量子力学...", "通用知识问答"),
    ("你说得对", "谢谢认可。", "交互反馈"),
    ("Python的list和tuple有什么区别", "list可变，tuple不可变...", "编程问答"),
    ("再见", "再见，祝你好运！", "告别"),
]


async def test_intent(user_msg: str, assistant_msg: str, desc: str):
    print(f"\n{'='*60}")
    print(f"测试: [{desc}]")
    print(f"  用户: {user_msg}")
    print(f"  AI: {assistant_msg}")
    print(f"{'='*60}")

    prompt = PROMPT_TEMPLATE.format(
        user_message=user_msg,
        assistant_message=assistant_msg,
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{SILICONFLOW_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": SILICONFLOW_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "stream": False,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  Raw output: {content}")

            try:
                parsed = json.loads(content)
                print(f"  Parsed JSON: {json.dumps(parsed, ensure_ascii=False, indent=2)}")
                needs = parsed.get("needs_long_term_memory", False)
                mem = parsed.get("memory_content", "")
                print(f"  -> needs_long_term_memory: {needs}")
                print(f"  -> memory_content: {mem}")

                if needs and mem:
                    print(f"  [STORE] Memory would be stored!")
                elif not needs:
                    print(f"  [SKIP] No personal info detected - correctly skipped")
                else:
                    print(f"  [WARN] needs=true but memory_content is empty")
            except json.JSONDecodeError as e:
                print(f"  [FAIL] JSON parse error: {e}")
    except Exception as e:
        print(f"  [ERROR] {e}")


async def main():
    print("=" * 60)
    print("  记忆意图识别模型输出诊断")
    print(f"  API: SiliconFlow")
    print(f"  Model: {SILICONFLOW_MODEL}")
    print(f"  Base URL: {SILICONFLOW_BASE_URL}")
    print("=" * 60)

    positive = 0
    negative = 0
    for user_msg, assistant_msg, desc in test_cases:
        await test_intent(user_msg, assistant_msg, desc)
        if any(kw in desc for kw in ["搜索", "打开", "问候", "致谢", "翻译", "写文档", "问答", "交互", "告别"]):
            negative += 1
        else:
            positive += 1

    print(f"\n{'='*60}")
    print(f"  诊断完成")
    print(f"  正样本（应记忆）: {positive} 个")
    print(f"  负样本（不应记忆）: {negative} 个")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
