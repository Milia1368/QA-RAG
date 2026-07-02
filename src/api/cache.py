"""
Redis 缓存层
对相同 (query, mode) 的请求进行缓存，避免重复推理，降低 P99 延迟。
缓存 key 基于 SHA256(query + mode) 生成，TTL 默认 1 小时。
"""

import hashlib
import json
from typing import Optional

import redis
from loguru import logger


class RAGCache:
    """
    封装 Redis 缓存操作，提供 get / set / invalidate 接口。
    连接失败时自动降级为无缓存模式（不影响主流程）。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str = "",
        ttl: int = 3600,
        max_connections: int = 20,
    ):
        self.ttl = ttl
        self._client: Optional[redis.Redis] = None
        self._available = False

        try:
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password or None,
                max_connections=max_connections,
                decode_responses=True,
            )
            self._client = redis.Redis(connection_pool=pool)
            self._client.ping()
            self._available = True
            logger.info(f"Redis 连接成功: {host}:{port}/{db}")
        except Exception as e:
            logger.warning(f"Redis 连接失败，将以无缓存模式运行: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    @staticmethod
    def _make_key(query: str, mode: str) -> str:
        """生成缓存 key：rag:cache:<sha256(query+mode)>。"""
        raw = f"{query.strip().lower()}|{mode}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"rag:cache:{digest}"

    def get(self, query: str, mode: str) -> Optional[dict]:
        """
        获取缓存的问答结果。

        Returns:
            缓存命中时返回 dict，未命中或不可用时返回 None
        """
        if not self._available:
            return None
        try:
            key = self._make_key(query, mode)
            cached = self._client.get(key)
            if cached:
                logger.debug(f"[Cache] 命中: key={key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"[Cache] 读取失败: {e}")
        return None

    def set(self, query: str, mode: str, result: dict) -> bool:
        """
        写入缓存。

        Args:
            query: 用户问题
            mode: 检索策略
            result: 待缓存的结果字典

        Returns:
            是否写入成功
        """
        if not self._available:
            return False
        try:
            key = self._make_key(query, mode)
            self._client.setex(key, self.ttl, json.dumps(result, ensure_ascii=False))
            logger.debug(f"[Cache] 写入: key={key}, ttl={self.ttl}s")
            return True
        except Exception as e:
            logger.warning(f"[Cache] 写入失败: {e}")
            return False

    def invalidate(self, query: str, mode: str) -> bool:
        """手动失效指定缓存。"""
        if not self._available:
            return False
        try:
            key = self._make_key(query, mode)
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"[Cache] 失效操作失败: {e}")
            return False
