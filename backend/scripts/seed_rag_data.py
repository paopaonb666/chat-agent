"""
RAG 系统压测数据生成脚本。

功能：
    - 在 PostgreSQL 中创建测试用户、对话、消息
    - 为每条消息生成 embedding 并写入 Milvus (conversation_history)
    - 同时生成用户记忆并写入 mem0_memories
    - 支持并发加速 embedding 生成

用法：
    cd backend && venv\\Scripts\\python scripts\\seed_rag_data.py
    cd backend && venv\\Scripts\\python scripts\\seed_rag_data.py --conversations 200 --messages-per-conv 50
    cd backend && venv\\Scripts\\python scripts\\seed_rag_data.py --clean-first

环境要求：
    - PostgreSQL 运行中
    - Milvus 运行中
    - Ollama 运行中且已拉取 embedding 模型
"""
import argparse
import asyncio
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

try:
    from pymilvus import MilvusClient
except ImportError:
    print("错误：未在虚拟环境中运行！请使用 venv\\Scripts\\python 执行")
    sys.exit(1)

env_path = os.path.join(base_dir, ".env")
if os.path.exists(env_path):
    from dotenv import load_dotenv
    load_dotenv(env_path)

import httpx
import jieba
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SessionLocal, engine
from app.models import User, Conversation, Message, UserMemory
from app.core.config import settings
from app.services.milvus_store import get_milvus_client, ensure_collection, COLLECTION_NAME, insert_message

# ============== 配置 ==============
OLLAMA_EMBED_URL = f"{settings.ollama_base_url}/api/embed"
EMBEDDING_MODEL = settings.embedding_model
EMBEDDING_DIM = 1024
BATCH_SIZE = 64
MAX_WORKERS = 4

# ============== 中文对话语料库 ==============
TOPICS = [
    "人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理",
    "计算机视觉", "推荐系统", "数据挖掘", "大数据分析", "云计算",
    "Docker", "Kubernetes", "微服务架构", "DevOps", "CI/CD",
    "Python编程", "Java开发", "Go语言", "Rust", "TypeScript",
    "React", "Vue", "Angular", "前端性能优化", "Web安全",
    "数据库优化", "Redis缓存", "消息队列", "Elasticsearch", "MongoDB",
    "投资理财", "股票基金", "房地产市场", "保险规划", "退休计划",
    "健康饮食", "运动健身", "瑜伽冥想", "睡眠质量", "心理健康",
    "旅行攻略", "日本旅游", "欧洲自由行", "东南亚", "国内自驾游",
    "育儿经验", "亲子教育", "学区房", "兴趣班", "高考志愿",
    "职场发展", "面试技巧", "简历优化", "跳槽策略", "升职加薪",
    "人际关系", "沟通技巧", "情绪管理", "时间管理", "效率工具",
    "读书笔记", "科幻小说", "历史传记", "心理学", "哲学思考",
    "音乐欣赏", "电影评论", "追剧讨论", "游戏攻略", "摄影技巧",
    "美食探店", "烘焙教程", "咖啡文化", "茶道", "品酒",
    "宠物养护", "猫狗训练", "植物种植", "园艺设计", "家居装修",
    "汽车选购", "电动车", "智能驾驶", "保养维修", "驾驶技巧",
    "法律常识", "劳动合同", "消费者权益", "知识产权", "税务筹划",
    "医疗常识", "体检报告", "慢性病管理", "中医养生", "疫苗接种",
]

USER_INTENTS = [
    "什么是{topic}？能给我简单介绍一下吗？",
    "我想学习{topic}，有什么好的入门资源推荐？",
    "我在做{topic}相关的项目，遇到了一些问题...",
    "你对{topic}怎么看？最近有什么新进展吗？",
    "请帮我总结一下{topic}的核心要点",
    "{topic}和{topic2}有什么区别？",
    "我在准备{topic}的面试，有什么建议？",
    "能推荐几本关于{topic}的好书吗？",
    "{topic}在实际工作中怎么应用？",
    "帮我写一个关于{topic}的教程大纲",
]

ASSISTANT_RESPONSES = [
    "{topic}是一个非常有意思的领域。简单来说，它主要是指...\n\n从发展历程来看，{topic}经历了几个重要阶段...\n\n如果你想深入学习，我建议从以下几个方面入手：\n1. 基础概念理解\n2. 核心原理掌握\n3. 实践项目练习\n4. 社区交流讨论",
    "关于{topic}，我为你整理了一些优质资源：\n\n**在线课程**\n- Coursera 上的相关专项课程\n- B站上的中文教程\n\n**书籍推荐**\n- 入门：《{topic}入门指南》\n- 进阶：《深入理解{topic}》\n\n**实践平台**\n- GitHub 上的开源项目\n- Kaggle 相关竞赛",
    "你提到的{topic}项目问题，我来帮你分析一下。\n\n常见的问题和解决方案：\n\n1. **性能瓶颈**\n   - 检查算法复杂度\n   - 考虑使用缓存机制\n   - 优化数据库查询\n\n2. **架构设计**\n   - 模块化拆分\n   - 接口抽象\n   - 异常处理完善\n\n3. **测试覆盖**\n   - 单元测试\n   - 集成测试\n   - 性能测试",
    "最近{topic}领域确实有不少新动态：\n\n- 新的研究成果不断涌出\n- 工业界应用越来越广泛\n- 工具和框架持续迭代\n\n值得关注的发展趋势：\n1. 与AI技术的深度融合\n2. 云原生架构支持\n3. 低代码/无代码化\n4. 安全性和隐私保护增强",
    "{topic}的核心要点可以总结为：\n\n**核心概念**\n- 定义与范畴\n- 基本原理\n- 关键指标\n\n**技术栈**\n- 基础工具\n- 主流框架\n- 辅助生态\n\n**最佳实践**\n- 设计模式\n- 编码规范\n- 部署策略",
    "{topic}和{topic2}的主要区别：\n\n| 维度 | {topic} | {topic2} |\n|------|---------|----------|\n| 定位 | 侧重A方面 | 侧重B方面 |\n| 适用场景 | 场景X | 场景Y |\n| 学习曲线 | 较平缓 | 较陡峭 |\n| 社区生态 | 成熟丰富 | 快速发展 |\n\n选择建议：根据你的具体需求来决定...",
    "{topic}面试准备建议：\n\n**基础知识**\n- 核心概念要清晰\n- 常见算法要能手写\n- 时间空间复杂度分析\n\n**项目经验**\n- 准备2-3个相关项目\n- 能讲清楚技术选型原因\n- 突出解决的问题和成果\n\n**软实力**\n- 沟通表达能力\n- 团队协作经验\n- 学习成长潜力",
    "关于{topic}的书籍，我推荐这几本：\n\n📚 **入门**\n《{topic}基础教程》- 通俗易懂，适合零基础\n\n📚 **进阶**\n《{topic}实战》- 案例丰富，贴近实际\n《{topic}设计模式》- 架构思维培养\n\n📚 **高级**\n《{topic}源码剖析》- 深入原理\n《{topic}性能优化》- 专家经验",
    "{topic}在实际工作中的应用非常广泛：\n\n**互联网公司**\n- 大型系统架构\n- 高并发处理\n- 数据驱动决策\n\n**传统行业**\n- 数字化转型\n- 流程自动化\n- 智能化升级\n\n**创业公司**\n- MVP快速验证\n- 技术债务管理\n- 团队技术建设",
    "好的，这是一个关于{topic}的教程大纲：\n\n## 第一章：认识{topic}\n- 背景与动机\n- 基本概念\n- 应用场景\n\n## 第二章：核心原理\n- 工作机制\n- 关键算法\n- 架构设计\n\n## 第三章：动手实践\n- 环境搭建\n- Hello World\n- 综合案例\n\n## 第四章：进阶话题\n- 性能优化\n- 最佳实践\n- 常见问题",
]

MEMORY_CONTENTS = [
    "用户正在学习{topic}，计划3个月内掌握基础",
    "用户对{topic}有浓厚兴趣，经常关注相关动态",
    "用户在工作中使用{topic}解决实际问题",
    "用户希望深入了解{topic}的底层原理",
    "用户正在准备{topic}相关的认证考试",
    "用户有{topic}项目经验，想进一步提升",
    "用户关注{topic}与{topic2}的结合应用",
    "用户喜欢用{topic}做 side project",
    "用户在团队中负责{topic}相关技术方向",
    "用户希望找到{topic}领域的工作机会",
]


def generate_conversation_topic() -> tuple[str, str]:
    """生成一组相关话题。"""
    topic = random.choice(TOPICS)
    topic2 = random.choice([t for t in TOPICS if t != topic])
    return topic, topic2


def generate_message_pair(topic: str, topic2: str, turn: int) -> tuple[str, str]:
    """生成一轮对话（用户提问 + AI回复）。"""
    template_idx = turn % len(USER_INTENTS)
    user_msg = USER_INTENTS[template_idx].format(topic=topic, topic2=topic2)
    assistant_msg = ASSISTANT_RESPONSES[template_idx].format(topic=topic, topic2=topic2)
    return user_msg, assistant_msg


def generate_memory_content(topic: str, topic2: str) -> str:
    """生成用户记忆内容。"""
    template = random.choice(MEMORY_CONTENTS)
    return template.format(topic=topic, topic2=topic2)


# ============== Embedding 生成 ==============

def get_embedding_sync(texts: list[str]) -> list[list[float]]:
    """同步调用 Ollama 生成 embedding（用于线程池）。"""
    try:
        resp = httpx.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBEDDING_MODEL, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if embeddings and isinstance(embeddings[0], list):
            return embeddings
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], (int, float)):
            return [embeddings]
        raise ValueError(f"Unexpected format: {data}")
    except Exception as e:
        print(f"[WARN] Embedding batch failed: {e}, returning zero vectors")
        return [[0.0] * EMBEDDING_DIM for _ in texts]


async def get_embeddings_async(texts: list[str]) -> list[list[float]]:
    """异步生成 embedding。"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        return await loop.run_in_executor(pool, get_embedding_sync, texts)


# ============== 数据插入 ==============

def create_test_user(db: Session) -> User:
    """创建或获取测试用户。"""
    user = db.query(User).filter(User.username == "rag_test_user").first()
    if user:
        return user
    user = User(username="rag_test_user", password_hash="test_hash")
    db.add(user)
    db.commit()
    db.refresh(user)
    print(f"创建测试用户: id={user.id}")
    return user


def insert_conversations(db: Session, user: User, count: int) -> list[Conversation]:
    """批量创建对话。"""
    conversations = []
    now = datetime.now(timezone.utc)
    for i in range(count):
        topic, _ = generate_conversation_topic()
        conv = Conversation(
            user_id=user.id,
            title=f"关于{topic}的讨论 #{i+1}",
            model="deepseek-chat",
            created_at=now - timedelta(days=random.randint(0, 30)),
            updated_at=now,
        )
        conversations.append(conv)
    db.add_all(conversations)
    db.commit()
    for c in conversations:
        db.refresh(c)
    print(f"创建 {count} 个对话")
    return conversations


def insert_messages(db: Session, conversations: list[Conversation], messages_per_conv: int) -> list[Message]:
    """批量创建消息。"""
    all_messages = []
    for conv in conversations:
        topic, topic2 = generate_conversation_topic()
        for turn in range(messages_per_conv // 2):
            user_msg, assistant_msg = generate_message_pair(topic, topic2, turn)
            msg_user = Message(
                conversation_id=conv.id,
                role="user",
                content=user_msg,
                created_at=conv.created_at + timedelta(minutes=turn * 2),
            )
            msg_assistant = Message(
                conversation_id=conv.id,
                role="assistant",
                content=assistant_msg,
                created_at=conv.created_at + timedelta(minutes=turn * 2 + 1),
            )
            all_messages.extend([msg_user, msg_assistant])

    # 分批插入避免内存过大
    batch_size = 1000
    for i in range(0, len(all_messages), batch_size):
        db.add_all(all_messages[i:i+batch_size])
        db.commit()
    print(f"创建 {len(all_messages)} 条消息")
    return all_messages


def insert_memories_pg(db: Session, user: User, conversations: list[Conversation]) -> list[UserMemory]:
    """在 PostgreSQL 中创建用户记忆。"""
    memories = []
    for conv in conversations:
        if random.random() < 0.3:  # 30% 的对话生成记忆
            topic, topic2 = generate_conversation_topic()
            mem = UserMemory(
                user_id=user.id,
                content=generate_memory_content(topic, topic2),
                source="auto_extracted",
                created_at=conv.created_at,
            )
            memories.append(mem)
    db.add_all(memories)
    db.commit()
    print(f"创建 {len(memories)} 条用户记忆 (PostgreSQL)")
    return memories


async def insert_milvus_messages(client: MilvusClient, messages: list[Message], user_id: int) -> None:
    """为消息生成 embedding 并写入 Milvus。"""
    ensure_collection(client, dim=EMBEDDING_DIM)

    total = len(messages)
    print(f"开始为 {total} 条消息生成 embedding 并写入 Milvus...")

    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch = messages[i:i+BATCH_SIZE]
        texts = [m.content for m in batch]

        embeddings = await get_embeddings_async(texts)

        for msg, emb in zip(batch, embeddings):
            insert_message(
                client,
                conversation_id=msg.conversation_id,
                user_id=user_id,
                role=msg.role,
                content=msg.content,
                message_id=msg.id,
                dense_embedding=emb,
                timestamp=int(msg.created_at.timestamp()),
            )

        inserted += len(batch)
        if (i // BATCH_SIZE + 1) % 10 == 0 or inserted >= total:
            print(f"  Milvus 进度: {inserted}/{total} ({inserted*100//total}%)")

    print(f"Milvus 写入完成: {inserted} 条")


async def insert_mem0_memories(client: MilvusClient, memories: list[UserMemory], user_id: int) -> None:
    """为用户记忆生成 embedding 并写入 mem0_memories collection。"""
    coll_name = "mem0_memories"

    # 确保 collection 存在（复用 conversation_history 的 schema 结构）
    if coll_name not in client.list_collections():
        from pymilvus import DataType
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("user_id", DataType.INT64)
        schema.add_field("memory_id", DataType.INT64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("timestamp", DataType.INT64)
        client.create_collection(collection_name=coll_name, schema=schema)
        idx = client.prepare_index_params()
        idx.add_index(
            field_name="dense_embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        client.create_index(collection_name=coll_name, index_params=idx)

    total = len(memories)
    if total == 0:
        print("没有用户记忆需要写入 mem0_memories")
        return

    print(f"开始为 {total} 条记忆生成 embedding 并写入 mem0_memories...")

    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch = memories[i:i+BATCH_SIZE]
        texts = [m.content for m in batch]
        embeddings = await get_embeddings_async(texts)

        data = []
        for mem, emb in zip(batch, embeddings):
            data.append({
                "user_id": user_id,
                "memory_id": mem.id,
                "content": mem.content,
                "dense_embedding": emb,
                "timestamp": int(mem.created_at.timestamp()),
            })

        client.insert(collection_name=coll_name, data=data)
        inserted += len(batch)
        if (i // BATCH_SIZE + 1) % 5 == 0 or inserted >= total:
            print(f"  mem0_memories 进度: {inserted}/{total} ({inserted*100//total}%)")

    print(f"mem0_memories 写入完成: {inserted} 条")


# ============== 清理 ==============

def clean_existing_data(db: Session, user_id: int) -> None:
    """清理该测试用户的已有数据。"""
    print("清理已有测试数据...")

    # 获取该用户的所有对话ID
    conv_ids = db.execute(
        text("SELECT id FROM conversations WHERE user_id = :uid"),
        {"uid": user_id}
    ).scalars().all()

    if conv_ids:
        # 删除消息
        db.execute(
            text("DELETE FROM messages WHERE conversation_id = ANY(:cids)"),
            {"cids": conv_ids}
        )
        # 删除文件
        db.execute(
            text("DELETE FROM uploaded_files WHERE conversation_id = ANY(:cids)"),
            {"cids": conv_ids}
        )
        # 删除对话
        db.execute(
            text("DELETE FROM conversations WHERE user_id = :uid"),
            {"uid": user_id}
        )

    # 删除用户记忆
    db.execute(
        text("DELETE FROM user_memories WHERE user_id = :uid"),
        {"uid": user_id}
    )

    db.commit()
    print(f"  PostgreSQL 数据已清理")

    # 清理 Milvus
    try:
        client = get_milvus_client()
        for coll in [COLLECTION_NAME, "mem0_memories"]:
            if coll in client.list_collections():
                # 删除该用户的数据
                client.delete(collection_name=coll, filter=f"user_id == {user_id}")
        print(f"  Milvus 数据已清理")
    except Exception as e:
        print(f"  [WARN] Milvus 清理失败: {e}")


# ============== 主流程 ==============

async def main():
    parser = argparse.ArgumentParser(description="RAG 系统压测数据生成")
    parser.add_argument("--conversations", type=int, default=100, help="对话数量 (默认100)")
    parser.add_argument("--messages-per-conv", type=int, default=100, help="每对话消息数 (默认100)")
    parser.add_argument("--clean-first", action="store_true", help="先清理已有测试数据")
    args = parser.parse_args()

    total_messages = args.conversations * args.messages_per_conv
    print("=" * 60)
    print("  RAG 系统压测数据生成")
    print("=" * 60)
    print(f"配置:")
    print(f"  对话数量: {args.conversations}")
    print(f"  每对话消息数: {args.messages_per_conv}")
    print(f"  总消息数: {total_messages}")
    print(f"  Embedding 模型: {EMBEDDING_MODEL}")
    print(f"  Embedding 维度: {EMBEDDING_DIM}")
    print(f"  批量大小: {BATCH_SIZE}")
    print(f"  并发数: {MAX_WORKERS}")
    print("=" * 60)

    # 检查服务状态
    print("\n检查服务状态...")
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        print("  PostgreSQL: OK")
    except Exception as e:
        print(f"  PostgreSQL: 失败 - {e}")
        sys.exit(1)
    finally:
        db.close()

    try:
        client = get_milvus_client()
        client.list_collections()
        print("  Milvus: OK")
    except Exception as e:
        print(f"  Milvus: 失败 - {e}")
        sys.exit(1)

    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if EMBEDDING_MODEL not in models and not any(EMBEDDING_MODEL in m for m in models):
            print(f"  [WARN] Ollama 中可能未找到模型 {EMBEDDING_MODEL}")
            print(f"  可用模型: {models}")
        else:
            print(f"  Ollama: OK (找到 {EMBEDDING_MODEL})")
    except Exception as e:
        print(f"  Ollama: 失败 - {e}")
        sys.exit(1)

    # 开始生成数据
    print("\n开始生成数据...")
    start_time = time.time()

    db = SessionLocal()
    try:
        # 1. 创建测试用户
        user = create_test_user(db)
        user_id = user.id  # 保存 user_id，避免后续访问分离的对象

        # 2. 清理已有数据（可选）
        if args.clean_first:
            clean_existing_data(db, user_id)

        # 3. 创建对话
        conversations = insert_conversations(db, user, args.conversations)

        # 4. 创建消息
        messages = insert_messages(db, conversations, args.messages_per_conv)

        # 5. 创建用户记忆 (PostgreSQL)
        memories = insert_memories_pg(db, user, conversations)

    finally:
        db.close()

    pg_time = time.time()
    print(f"\nPostgreSQL 数据插入完成，耗时: {pg_time - start_time:.1f}s")

    # 6. 写入 Milvus (消息 embedding)
    client = get_milvus_client()
    await insert_milvus_messages(client, messages, user_id)

    milvus_time = time.time()
    print(f"Milvus 消息写入完成，耗时: {milvus_time - pg_time:.1f}s")

    # 7. 写入 mem0_memories
    await insert_mem0_memories(client, memories, user_id)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  数据生成完成！")
    print("=" * 60)
    print(f"统计:")
    print(f"  用户: rag_test_user (id={user_id})")
    print(f"  对话: {len(conversations)}")
    print(f"  消息: {len(messages)}")
    print(f"  用户记忆: {len(memories)}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  平均每条消息: {total_time/len(messages)*1000:.0f}ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
