"""
End-to-end memory chain validation test.
Tests: rule-based detection + model intent → memory_prompt → store endpoint.
"""
import asyncio
import json
import sys
import os

# Fix Windows console encoding for UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from app.services.intent import detect_personal_info, recognize_intent, IntentResult
from app.services.embedding import get_dense_embedding
from app.services.milvus_store import get_milvus_client, ensure_collection, insert_message, COLLECTION_NAME


async def main():
    print("=" * 60)
    print("  端到端记忆链路验证")
    print("=" * 60)

    test_cases = [
        ("我叫炮炮", "自我介绍"),
        ("我住在北京朝阳区", "住址"),
        ("我的职业是程序员", "职业"),
        ("请记住我的名字叫小明", "显式记忆请求"),
        ("今天天气怎么样", "非个人信息-天气"),
        ("你好", "非个人信息-问候"),
    ]

    # ── 1. Rule-based detection ──
    print("\n[1] 规则匹配检测：")
    all_ok = True
    for query, desc in test_cases:
        has_memory, content = detect_personal_info(query)
        status = "DETECT" if has_memory else "pass"
        marker = "" if has_memory or desc.startswith("非个人信息") else " ⚠️ MISS!"
        print(f"  [{status}] {desc}: '{query}' → {has_memory} | {content[:50]}{marker}")
        if marker:
            all_ok = False

    # ── 2. Model intent (for comparison) ──
    print("\n[2] 模型意图判断（对比）：")
    personal_queries = [q for q, d in test_cases if not d.startswith("非个人信息")]
    for query in personal_queries:
        try:
            intent = await recognize_intent(query, [])
            print(f"  query='{query}': model needs_long_term_memory={intent.needs_long_term_memory}, memory_content='{intent.memory_content[:50]}'")
        except Exception as e:
            print(f"  query='{query}': model ERROR: {e}")

    # ── 3. Memory store chain ──
    print("\n[3] 记忆存储链路（Milvus 写入测试）：")
    test_content = "我的名字叫炮炮"
    try:
        vec = await get_dense_embedding(test_content)
        print(f"  向量维度: {len(vec)} ✓")

        client = get_milvus_client()
        ensure_collection(client)

        import time
        insert_message(
            client,
            conversation_id="test-diag",
            user_id=-1,
            role="memory",
            content=test_content,
            message_id=-1,
            dense_embedding=vec,
            timestamp=int(time.time()),
        )
        stats = client.get_collection_stats(COLLECTION_NAME)
        print(f"  Milvus row_count: {stats.get('row_count', 'unknown')} ✓")
        print(f"  存储成功: '{test_content}' as role='memory'")
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  全部检测通过")
    else:
        print("  有未检测到的情况（标记 ⚠️）")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
