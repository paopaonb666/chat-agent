"""
RAG 系统性能测试脚本。

测试范围：
    1. Embedding 生成性能
    2. Milvus Dense Search 性能
    3. PostgreSQL BM25 候选获取性能
    4. Hybrid Search (Dense + BM25 + RRF) 性能
    5. Rerank 性能
    6. 完整 RAG Pipeline 端到端性能

用法：
    cd backend && venv\\Scripts\\python scripts\\benchmark_rag.py
    cd backend && venv\\Scripts\\python scripts\\benchmark_rag.py --queries 50 --user-id 1
    cd backend && venv\\Scripts\\python scripts\\benchmark_rag.py --skip-rerank
    cd backend && venv\\Scripts\\python scripts\\benchmark_rag.py --output result.json

环境要求：
    - PostgreSQL 运行中且有数据
    - Milvus 运行中且有数据
    - Ollama 运行中
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime

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

from app.db import SessionLocal
from app.models import User, Message, Conversation
from app.services.embedding import get_dense_embedding
from app.services.milvus_store import get_milvus_client, search_dense
from app.services.hybrid_search import hybrid_search, _fetch_candidates_from_pg, _bm25_search, _reciprocal_rank_fusion
from app.services.rerank import rerank_passages
from app.services.rag_pipeline import run_rag


# ============== 测试查询集 ==============
TEST_QUERIES = [
    "什么是深度学习？",
    "推荐系统怎么实现？",
    "Docker 和 Kubernetes 有什么区别？",
    "Python 异步编程最佳实践",
    "如何优化数据库查询性能？",
    "微服务架构的优缺点",
    "Redis 缓存穿透和雪崩怎么解决？",
    "前端性能优化有哪些方法？",
    "机器学习模型部署方案",
    "CI/CD 流水线设计",
    "什么是自然语言处理？",
    "云计算和传统服务器区别",
    "Go 语言适合做什么？",
    "React 和 Vue 怎么选？",
    "数据挖掘常用算法",
    "如何学习投资理财？",
    "健康饮食有什么建议？",
    "日本旅游攻略推荐",
    "育儿经验分享",
    "职场晋升技巧",
    "时间管理方法",
    "心理学入门书籍",
    "摄影技巧学习",
    "美食探店推荐",
    "宠物养护知识",
    "汽车选购指南",
    "法律常识科普",
    "中医养生方法",
    "瑜伽入门教程",
    "咖啡文化介绍",
]


@dataclass
class BenchmarkResult:
    name: str
    queries: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    results_per_query: int = 0


def print_result(r: BenchmarkResult):
    print(f"\n  [{r.name}]")
    print(f"    查询次数: {r.queries}")
    print(f"    总耗时: {r.total_time_ms:.1f}ms ({r.total_time_ms/1000:.1f}s)")
    print(f"    平均: {r.avg_time_ms:.1f}ms")
    print(f"    最小: {r.min_time_ms:.1f}ms")
    print(f"    最大: {r.max_time_ms:.1f}ms")
    print(f"    中位数: {r.median_time_ms:.1f}ms")
    print(f"    P95: {r.p95_time_ms:.1f}ms")
    print(f"    P99: {r.p99_time_ms:.1f}ms")
    if r.results_per_query > 0:
        print(f"    每查询结果数: {r.results_per_query}")


def calc_stats(times_ms: list[float], name: str, results_per_query: int = 0) -> BenchmarkResult:
    sorted_times = sorted(times_ms)
    n = len(sorted_times)
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)
    return BenchmarkResult(
        name=name,
        queries=n,
        total_time_ms=sum(sorted_times),
        avg_time_ms=statistics.mean(sorted_times),
        min_time_ms=min(sorted_times),
        max_time_ms=max(sorted_times),
        median_time_ms=statistics.median(sorted_times),
        p95_time_ms=sorted_times[min(p95_idx, n - 1)],
        p99_time_ms=sorted_times[min(p99_idx, n - 1)],
        results_per_query=results_per_query,
    )


# ============== 各阶段测试 ==============

async def benchmark_embedding(queries: list[str]) -> BenchmarkResult:
    """测试 Embedding 生成性能。"""
    times = []
    for q in queries:
        start = time.perf_counter()
        await get_dense_embedding(q)
        times.append((time.perf_counter() - start) * 1000)
    return calc_stats(times, "Embedding 生成")


def benchmark_milvus_dense(client, queries: list[str], embeddings: list[list[float]], user_id: int | None) -> BenchmarkResult:
    """测试 Milvus Dense Search 性能。"""
    times = []
    result_counts = []
    for emb in embeddings:
        start = time.perf_counter()
        results = search_dense(client, emb, user_id, top_k=30)
        times.append((time.perf_counter() - start) * 1000)
        result_counts.append(len(results))
    return calc_stats(times, "Milvus Dense Search", int(statistics.mean(result_counts)))


def benchmark_pg_candidates(user_id: int | None) -> BenchmarkResult:
    """测试 PostgreSQL 候选获取性能。"""
    times = []
    result_counts = []
    for _ in range(20):
        start = time.perf_counter()
        candidates = _fetch_candidates_from_pg(user_id, limit=200)
        times.append((time.perf_counter() - start) * 1000)
        result_counts.append(len(candidates))
    return calc_stats(times, "PostgreSQL 候选获取", int(statistics.mean(result_counts)))


def benchmark_bm25(queries: list[str], candidates: list[dict]) -> BenchmarkResult:
    """测试 BM25 检索性能。"""
    times = []
    result_counts = []
    for q in queries:
        start = time.perf_counter()
        results = _bm25_search(q, candidates, top_k=30)
        times.append((time.perf_counter() - start) * 1000)
        result_counts.append(len(results))
    return calc_stats(times, "BM25 检索", int(statistics.mean(result_counts)))


async def benchmark_hybrid(queries: list[str], embeddings: list[list[float]], user_id: int | None) -> BenchmarkResult:
    """测试 Hybrid Search 完整流程性能。"""
    times = []
    result_counts = []
    for q, emb in zip(queries, embeddings):
        start = time.perf_counter()
        results = await hybrid_search(q, emb, user_id, top_k=15)
        times.append((time.perf_counter() - start) * 1000)
        result_counts.append(len(results))
    return calc_stats(times, "Hybrid Search (Dense+BM25+RRF)", int(statistics.mean(result_counts)))


async def benchmark_rerank(queries: list[str], hybrid_results_list: list[list[dict]]) -> BenchmarkResult:
    """测试 Rerank 性能。"""
    times = []
    result_counts = []
    for q, results in zip(queries, hybrid_results_list):
        if len(results) < 3:
            continue
        start = time.perf_counter()
        reranked = await rerank_passages(q, results, top_n=5)
        times.append((time.perf_counter() - start) * 1000)
        result_counts.append(len(reranked))
    if not times:
        return calc_stats([0], "Rerank (跳过，结果太少)")
    return calc_stats(times, "Rerank", int(statistics.mean(result_counts)))


async def benchmark_full_pipeline(queries: list[str], user_id: int | None, skip_rerank: bool) -> BenchmarkResult:
    """测试完整 RAG Pipeline 端到端性能。"""
    times = []
    result_lengths = []
    for q in queries:
        start = time.perf_counter()
        if skip_rerank:
            # 手动调用各阶段，跳过 rerank
            emb = await get_dense_embedding(q)
            hybrid_results = await hybrid_search(q, emb, user_id, top_k=15)
            context_lines = [f"[{i+1}] {r['content'][:100]}..." for i, r in enumerate(hybrid_results[:5])]
            result = "\n".join(context_lines)
        else:
            result = await run_rag(q, conversation_id="benchmark", user_id=user_id, messages=[])
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        result_lengths.append(len(result))
    name = "完整 RAG Pipeline (无 Rerank)" if skip_rerank else "完整 RAG Pipeline (含 Rerank)"
    r = calc_stats(times, name)
    r.results_per_query = int(statistics.mean(result_lengths))
    return r


# ============== 主流程 ==============

async def main():
    parser = argparse.ArgumentParser(description="RAG 系统性能测试")
    parser.add_argument("--queries", type=int, default=30, help="测试查询数量 (默认30)")
    parser.add_argument("--user-id", type=int, default=None, help="指定用户ID测试 (默认不指定)")
    parser.add_argument("--skip-rerank", action="store_true", help="跳过 Rerank 阶段测试")
    parser.add_argument("--output", type=str, default=None, help="输出结果到 JSON 文件")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "embedding", "milvus", "pg", "bm25", "hybrid", "rerank", "pipeline"],
                        help="指定测试阶段")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG 系统性能测试")
    print("=" * 60)
    print(f"配置:")
    print(f"  测试查询数: {args.queries}")
    print(f"  用户ID: {args.user_id if args.user_id else '未指定 (全局搜索)'}")
    print(f"  跳过 Rerank: {args.skip_rerank}")
    print(f"  测试阶段: {args.stage}")
    print("=" * 60)

    # 准备查询
    queries = (TEST_QUERIES * ((args.queries // len(TEST_QUERIES)) + 1))[:args.queries]
    print(f"\n准备 {len(queries)} 条测试查询...")

    # 检查数据库状态
    print("\n检查数据库状态...")
    db = SessionLocal()
    try:
        msg_count = db.query(Message).count()
        conv_count = db.query(Conversation).count()
        user_count = db.query(User).count()
        print(f"  PostgreSQL: {conv_count} 对话, {msg_count} 消息, {user_count} 用户")

        if args.user_id:
            user_msg_count = db.query(Message).join(Conversation).filter(Conversation.user_id == args.user_id).count()
            print(f"  用户 {args.user_id} 的消息数: {user_msg_count}")
    finally:
        db.close()

    try:
        client = get_milvus_client()
        coll_stats = client.get_collection_stats("conversation_history")
        print(f"  Milvus conversation_history: {coll_stats.get('row_count', 'unknown')} 条")
    except Exception as e:
        print(f"  [WARN] Milvus 状态获取失败: {e}")
        client = None

    # 预生成所有 embedding（避免重复计算影响各阶段测试）
    print("\n预生成查询 embedding...")
    embeddings = []
    for i, q in enumerate(queries):
        emb = await get_dense_embedding(q)
        embeddings.append(emb)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(queries)}")
    print(f"  Embedding 维度: {len(embeddings[0])}")

    all_results = []

    # 1. Embedding 生成测试
    if args.stage in ("all", "embedding"):
        print("\n" + "-" * 40)
        print("测试 1: Embedding 生成性能")
        print("-" * 40)
        r = await benchmark_embedding(queries)
        print_result(r)
        all_results.append(asdict(r))

    # 2. Milvus Dense Search
    if args.stage in ("all", "milvus") and client:
        print("\n" + "-" * 40)
        print("测试 2: Milvus Dense Search 性能")
        print("-" * 40)
        r = benchmark_milvus_dense(client, queries, embeddings, args.user_id)
        print_result(r)
        all_results.append(asdict(r))

    # 3. PostgreSQL 候选获取
    if args.stage in ("all", "pg"):
        print("\n" + "-" * 40)
        print("测试 3: PostgreSQL 候选获取性能")
        print("-" * 40)
        r = benchmark_pg_candidates(args.user_id)
        print_result(r)
        all_results.append(asdict(r))

    # 4. BM25 检索
    if args.stage in ("all", "bm25"):
        print("\n" + "-" * 40)
        print("测试 4: BM25 检索性能")
        print("-" * 40)
        candidates = _fetch_candidates_from_pg(args.user_id, limit=200)
        print(f"  候选集大小: {len(candidates)}")
        r = benchmark_bm25(queries, candidates)
        print_result(r)
        all_results.append(asdict(r))

    # 5. Hybrid Search
    if args.stage in ("all", "hybrid"):
        print("\n" + "-" * 40)
        print("测试 5: Hybrid Search 完整流程性能")
        print("-" * 40)
        r = await benchmark_hybrid(queries, embeddings, args.user_id)
        print_result(r)
        all_results.append(asdict(r))

    # 6. Rerank
    if args.stage in ("all", "rerank") and not args.skip_rerank:
        print("\n" + "-" * 40)
        print("测试 6: Rerank 性能")
        print("-" * 40)
        hybrid_results_list = []
        for q, emb in zip(queries, embeddings):
            results = await hybrid_search(q, emb, args.user_id, top_k=15)
            hybrid_results_list.append(results)
        r = await benchmark_rerank(queries, hybrid_results_list)
        print_result(r)
        all_results.append(asdict(r))

    # 7. 完整 Pipeline
    if args.stage in ("all", "pipeline"):
        print("\n" + "-" * 40)
        print("测试 7: 完整 RAG Pipeline 端到端性能")
        print("-" * 40)
        r = await benchmark_full_pipeline(queries, args.user_id, args.skip_rerank)
        print_result(r)
        all_results.append(asdict(r))

    # 汇总
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    pipeline_result = None
    for r in all_results:
        name = r["name"]
        avg = r["avg_time_ms"]
        p95 = r["p95_time_ms"]
        print(f"  {name:<40} avg={avg:>8.1f}ms  p95={p95:>8.1f}ms")
        if "完整 RAG Pipeline" in name:
            pipeline_result = r

    if pipeline_result:
        qps = 1000.0 / pipeline_result["avg_time_ms"]
        print(f"\n  预估 QPS (基于完整 Pipeline): {qps:.2f}")

    # 输出到文件
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "queries": args.queries,
                "user_id": args.user_id,
                "skip_rerank": args.skip_rerank,
                "stage": args.stage,
            },
            "results": all_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存到: {args.output}")

    print("\n" + "=" * 60)
    print("  测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
