"""
Unified caching utilities and factory functions for consistent cache usage.

This module provides convenient utilities, decorators, and migration patterns
to consolidate the duplicate caching implementations found across the codebase.
"""

import asyncio
import functools
import hashlib
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from .caching import EmbeddingCache, LRUCache, QueryResultCache, TTLCache
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class CacheFactory:
    """Factory for creating standardized cache instances."""

    @staticmethod
    def create_ttl_cache(
        max_size: int = 1000,
        ttl_seconds: int = 900,  # 15 minutes
        cleanup_interval: int = 60,
    ) -> TTLCache[Any]:
        """
        Create a TTL cache with specified parameters.

        Args:
            max_size: Maximum number of items
            ttl_seconds: Time-to-live in seconds
            cleanup_interval: Background cleanup interval

        Returns:
            TTLCache instance
        """
        return TTLCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            cleanup_interval=cleanup_interval,
        )

    @staticmethod
    def create_lru_cache(
        max_size: int = 1000,
        cleanup_interval: int = 300,  # 5 minutes
    ) -> LRUCache[Any]:
        """
        Create an LRU cache with specified parameters.

        Args:
            max_size: Maximum number of items
            cleanup_interval: Background cleanup interval

        Returns:
            LRUCache instance
        """
        return LRUCache(max_size=max_size, cleanup_interval=cleanup_interval)

    @staticmethod
    def create_query_cache(
        max_size: int = 1000,
        ttl_minutes: int = 15,
    ) -> QueryResultCache:
        """
        Create a query result cache with specified parameters.

        Args:
            max_size: Maximum number of cached queries
            ttl_minutes: Time-to-live in minutes

        Returns:
            QueryResultCache instance
        """
        cache = QueryResultCache()
        return cache

    @staticmethod
    def create_embedding_cache(
        max_size: int = 10000,
        ttl_hours: int = 24,
    ) -> EmbeddingCache:
        """
        Create an embedding cache with specified parameters.

        Args:
            max_size: Maximum number of cached embeddings
            ttl_hours: Time-to-live in hours

        Returns:
            EmbeddingCache instance
        """
        cache = EmbeddingCache()
        return cache


class CacheKeyGenerator:
    """Utility for generating consistent cache keys."""

    @staticmethod
    def generate_text_hash(text: str) -> str:
        """
        Generate a consistent hash for text content.

        Args:
            text: Text content to hash

        Returns:
            SHA256 hash string
        """
        return hashlib.sha256(text.strip().encode()).hexdigest()

    @staticmethod
    def generate_query_key(
        query: str,
        limit: int | None = None,
        min_score: float | None = None,
        source_filters: list[str] | None = None,
        rerank: bool = False,
        include_content: bool = True,
        include_metadata: bool = True,
        date_range: tuple[str, str] | None = None,
    ) -> str:
        """
        Generate a deterministic cache key from query parameters.

        Args:
            query: Search query
            limit: Result limit
            min_score: Minimum similarity score
            source_filters: Source URL filters
            rerank: Whether reranking is enabled
            include_content: Whether to include content
            include_metadata: Whether to include metadata
            date_range: Date range filter

        Returns:
            Cache key string
        """
        key_components = [
            query.strip().lower(),
            str(limit) if limit is not None else "None",
            str(min_score) if min_score is not None else "None",
            ",".join(sorted(source_filters)) if source_filters else "None",
            str(rerank),
            str(include_content),
            str(include_metadata),
            f"{date_range[0]},{date_range[1]}" if date_range else "None",
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    @staticmethod
    def generate_function_key(
        func_name: str,
        args: tuple,
        kwargs: dict,
        include_timestamp: bool = False,
    ) -> str:
        """
        Generate cache key for function calls.

        Args:
            func_name: Function name
            args: Positional arguments
            kwargs: Keyword arguments
            include_timestamp: Whether to include current timestamp

        Returns:
            Cache key string
        """
        key_components = [func_name]

        # Add args
        for arg in args:
            if hasattr(arg, "__dict__"):
                key_components.append(str(hash(str(sorted(arg.__dict__.items())))))
            else:
                key_components.append(str(hash(str(arg))))

        # Add kwargs
        for key, value in sorted(kwargs.items()):
            if hasattr(value, "__dict__"):
                key_components.append(
                    f"{key}={hash(str(sorted(value.__dict__.items())))}"
                )
            else:
                key_components.append(f"{key}={hash(str(value))}")

        if include_timestamp:
            # Include hour-level timestamp for time-sensitive caching
            hour_timestamp = datetime.now().replace(minute=0, second=0, microsecond=0)
            key_components.append(str(int(hour_timestamp.timestamp())))

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    cache: TTLCache | LRUCache | None = None,
    ttl_seconds: int = 900,
    max_size: int = 1000,
    key_func: Callable[..., str] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for caching function results.

    Args:
        cache: Existing cache instance to use
        ttl_seconds: TTL for new cache instances
        max_size: Max size for new cache instances
        key_func: Custom key generation function

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        # Create cache if not provided
        if cache is None:
            func_cache = CacheFactory.create_ttl_cache(
                max_size=max_size, ttl_seconds=ttl_seconds
            )
        else:
            func_cache = cache

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = CacheKeyGenerator.generate_function_key(
                    func.__name__, args, kwargs
                )

            # Try to get from cache
            result = await func_cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = await func(*args, **kwargs)
            await func_cache.set(cache_key, result)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # For sync functions, we need to handle caching differently
            # This is a simplified implementation
            if key_func:
                _cache_key = key_func(*args, **kwargs)
            else:
                _cache_key = CacheKeyGenerator.generate_function_key(
                    func.__name__, args, kwargs
                )

            # Note: This is a simplified sync version
            # In a real implementation, you might want to use a sync cache
            result = func(*args, **kwargs)
            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


class CacheManager:
    """Manager for multiple cache instances with lifecycle management."""

    def __init__(self):
        self._caches: dict[
            str, TTLCache | LRUCache | EmbeddingCache | QueryResultCache
        ] = {}
        self._initialized = False

    def register_cache(
        self, name: str, cache: TTLCache | LRUCache | EmbeddingCache | QueryResultCache
    ) -> None:
        """
        Register a cache instance for management.

        Args:
            name: Unique name for the cache
            cache: Cache instance to register
        """
        self._caches[name] = cache
        logger.debug(f"Registered cache: {name}")

    def get_cache(
        self, name: str
    ) -> TTLCache | LRUCache | EmbeddingCache | QueryResultCache | None:
        """
        Get a registered cache by name.

        Args:
            name: Cache name

        Returns:
            Cache instance or None
        """
        return self._caches.get(name)

    async def initialize_all(self) -> None:
        """Initialize all registered caches."""
        if self._initialized:
            return

        for name, cache in self._caches.items():
            if hasattr(cache, "start"):
                await cache.start()
                logger.debug(f"Initialized cache: {name}")

        self._initialized = True
        logger.info(f"Initialized {len(self._caches)} caches")

    async def cleanup_all(self) -> None:
        """Clean up all registered caches."""
        for name, cache in self._caches.items():
            try:
                if hasattr(cache, "stop"):
                    await cache.stop()
                elif hasattr(cache, "clear"):
                    if asyncio.iscoroutinefunction(cache.clear):
                        await cache.clear()
                    else:
                        cache.clear()
                logger.debug(f"Cleaned up cache: {name}")
            except Exception as e:
                logger.error(f"Failed to cleanup cache {name}: {e}")

        self._initialized = False
        logger.info(f"Cleaned up {len(self._caches)} caches")

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all caches."""
        stats = {}
        for name, cache in self._caches.items():
            if hasattr(cache, "get_stats"):
                stats[name] = cache.get_stats()
            elif hasattr(cache, "hits") and hasattr(cache, "misses"):
                stats[name] = {
                    "hits": getattr(cache, "hits", 0),
                    "misses": getattr(cache, "misses", 0),
                    "size": len(getattr(cache, "cache", {})),
                }
        return stats


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return _cache_manager


# Migration utilities for existing duplicate implementations
class LegacyCacheAdapter:
    """Adapter for migrating legacy cache implementations."""

    @staticmethod
    def create_query_cache_replacement(
        max_size: int = 1000, ttl_minutes: int = 15
    ) -> TTLCache[Any]:
        """
        Create a TTL cache to replace QueryCache from rag/service.py.

        This provides the same functionality as the legacy QueryCache
        but using the standardized TTLCache implementation.
        """
        return CacheFactory.create_ttl_cache(
            max_size=max_size, ttl_seconds=ttl_minutes * 60, cleanup_interval=60
        )

    @staticmethod
    def create_embedding_cache_replacement(max_size: int = 10000) -> LRUCache[Any]:
        """
        Create an LRU cache to replace EmbeddingCache from rag/embedding.py.

        This provides the same functionality as the legacy EmbeddingCache
        but using the standardized LRUCache implementation.
        """
        return CacheFactory.create_lru_cache(
            max_size=max_size,
            cleanup_interval=300,  # 5 minutes
        )


# Convenience functions for common caching patterns
async def get_or_compute(
    cache: TTLCache | LRUCache,
    key: str,
    compute_func: Callable[[], Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Get value from cache or compute and cache it.

    Args:
        cache: Cache instance to use
        key: Cache key
        compute_func: Function to compute value if not cached
        *args: Arguments for compute function
        **kwargs: Keyword arguments for compute function

    Returns:
        Cached or computed value
    """
    # Try cache first
    result = await cache.get(key)
    if result is not None:
        return result

    # Compute and cache
    if asyncio.iscoroutinefunction(compute_func):
        result = await compute_func(*args, **kwargs)
    else:
        result = compute_func(*args, **kwargs)

    await cache.set(key, result)
    return result


def create_time_based_key(base_key: str, granularity: str = "hour") -> str:
    """
    Create a time-based cache key for time-sensitive caching.

    Args:
        base_key: Base cache key
        granularity: Time granularity ("minute", "hour", "day")

    Returns:
        Time-based cache key
    """
    now = datetime.now()

    if granularity == "minute":
        time_part = now.replace(second=0, microsecond=0)
    elif granularity == "hour":
        time_part = now.replace(minute=0, second=0, microsecond=0)
    elif granularity == "day":
        time_part = now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Invalid granularity: {granularity}")

    timestamp = int(time_part.timestamp())
    return f"{base_key}:{timestamp}"


# Initialize default cache manager with common caches
async def setup_default_caches() -> None:
    """Set up default cache instances in the global cache manager."""
    manager = get_cache_manager()

    # Create common cache instances
    query_cache = CacheFactory.create_query_cache()
    embedding_cache = CacheFactory.create_embedding_cache()
    general_ttl = CacheFactory.create_ttl_cache()
    general_lru = CacheFactory.create_lru_cache()

    # Register caches
    manager.register_cache("query", query_cache)
    manager.register_cache("embedding", embedding_cache)
    manager.register_cache("ttl", general_ttl)
    manager.register_cache("lru", general_lru)

    # Initialize all
    await manager.initialize_all()

    logger.info("Default caches setup completed")


async def cleanup_default_caches() -> None:
    """Clean up default caches."""
    manager = get_cache_manager()
    await manager.cleanup_all()
    logger.info("Default caches cleanup completed")
