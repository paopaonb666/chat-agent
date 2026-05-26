import logging
from arq import create_pool
from arq.connections import RedisSettings
from app.core.config import settings

logger = logging.getLogger(__name__)

redis_settings = RedisSettings.from_dsn(settings.redis_url)


async def get_arq_pool():
    return await create_pool(redis_settings)


class WorkerSettings:
    redis_settings = redis_settings
    functions = []

    @classmethod
    def register_function(cls, func):
        cls.functions.append(func)
        return func
