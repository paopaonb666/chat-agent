"""搜索引擎抽象层与降级管理器。

提供统一的 SearchEngine 接口和 FallbackSearchManager，
支持按优先级依次尝试多个搜索引擎，任一成功即返回。
"""

from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)


class SearchEngine(ABC):
    """搜索引擎抽象基类。"""

    name: str = ""

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[dict]:
        """执行同步搜索，返回统一格式的结果列表。

        Returns:
            list[dict]: 每个元素包含 title, url, snippet, position, engine
        """
        ...


class FallbackSearchManager:
    """按优先级依次尝试多个搜索引擎的降级管理器。"""

    def __init__(self, engines: list[SearchEngine]):
        self.engines = engines

    async def async_search(self, query: str, max_results: int) -> list[dict]:
        """异步依次尝试各引擎，任一成功即返回。"""
        for engine in self.engines:
            try:
                results = await asyncio.to_thread(engine.search, query, max_results)
                if results:
                    logger.info("Search succeeded via %s (%d results)", engine.name, len(results))
                    return results
            except Exception as exc:
                logger.warning("Engine %s failed: %s", engine.name, exc)
                continue

        logger.error("All search engines failed for query: %s", query)
        return []
