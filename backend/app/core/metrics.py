from prometheus_client import Counter, Histogram
from prometheus_client.registry import REGISTRY

_LLM_COUNTER_NAME = "llm_api_calls_total"
_llm_counter: Counter | None = None

_RAG_COUNTER_NAME = "rag_retrieval_total"
_rag_counter: Counter | None = None
_RAG_LATENCY_NAME = "rag_retrieval_duration_seconds"
_rag_latency: Histogram | None = None
_RAG_RESULT_NAME = "rag_result_count"
_rag_result: Histogram | None = None
_MEMORY_OP_NAME = "memory_operations_total"
_memory_op: Counter | None = None

_QUERY_REWRITE_NAME = "query_rewrites_total"
_query_rewrite_counter: Counter | None = None


def _get_or_create_metric(name: str, factory, *args, **kwargs):
    """Retrieve existing metric by name or create a new one."""
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return factory(*args, **kwargs)


def get_llm_counter() -> Counter:
    global _llm_counter
    if _llm_counter is not None:
        return _llm_counter
    _llm_counter = _get_or_create_metric(
        _LLM_COUNTER_NAME, Counter,
        _LLM_COUNTER_NAME, "Total LLM API calls", ["model", "status"],
    )
    return _llm_counter


def get_rag_retrieval_counter() -> Counter:
    global _rag_counter
    if _rag_counter is not None:
        return _rag_counter
    _rag_counter = _get_or_create_metric(
        _RAG_COUNTER_NAME, Counter,
        _RAG_COUNTER_NAME, "Total RAG retrieval calls", ["status"],
    )
    return _rag_counter


def get_rag_retrieval_latency() -> Histogram:
    global _rag_latency
    if _rag_latency is not None:
        return _rag_latency
    _rag_latency = _get_or_create_metric(
        _RAG_LATENCY_NAME, Histogram,
        _RAG_LATENCY_NAME, "RAG retrieval duration in seconds", ["status"],
    )
    return _rag_latency


def get_rag_result_count() -> Histogram:
    global _rag_result
    if _rag_result is not None:
        return _rag_result
    _rag_result = _get_or_create_metric(
        _RAG_RESULT_NAME, Histogram,
        _RAG_RESULT_NAME, "Number of RAG results per retrieval",
        ["stage"], buckets=[0, 1, 3, 5, 10, 20, 50],
    )
    return _rag_result


def get_memory_op_counter() -> Counter:
    global _memory_op
    if _memory_op is not None:
        return _memory_op
    _memory_op = _get_or_create_metric(
        _MEMORY_OP_NAME, Counter,
        _MEMORY_OP_NAME, "Total memory operations", ["operation", "status"],
    )
    return _memory_op


def get_query_rewrite_counter() -> Counter:
    global _query_rewrite_counter
    if _query_rewrite_counter is not None:
        return _query_rewrite_counter
    _query_rewrite_counter = _get_or_create_metric(
        _QUERY_REWRITE_NAME, Counter,
        _QUERY_REWRITE_NAME, "Total query rewrites", ["status"],
    )
    return _query_rewrite_counter
