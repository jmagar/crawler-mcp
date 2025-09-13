"""
Connection pooling for Qdrant client to improve performance.
"""

import asyncio
import logging
import random
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from ..settings import get_settings

logger = logging.getLogger(__name__)


class QdrantConnectionPool:
    """
    Connection pool for Qdrant clients with automatic health checking
    and connection recycling.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        size: int | None = None,
        timeout: float | None = None,
        health_check_interval: int = 60,
    ):
        """
        Initialize the connection pool.

        Args:
            url: Qdrant server URL
            api_key: API key for authentication
            size: Pool size (number of connections)
            timeout: Request timeout in seconds
            health_check_interval: Seconds between health checks
        """
        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.api_key = api_key or (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        )
        self.size = size or settings.qdrant_connection_pool_size
        self.timeout = timeout or settings.qdrant_timeout
        self.health_check_interval = health_check_interval

        # Connection pool
        self.connections: list[AsyncQdrantClient] = []
        self.available: asyncio.Queue[AsyncQdrantClient] = asyncio.Queue()
        self.connection_health: dict[AsyncQdrantClient, datetime] = {}

        # Statistics
        self.total_requests = 0
        self.failed_requests = 0
        self.recycled_connections = 0

        # Synchronization
        self._lock = asyncio.Lock()
        self._initialized = False
        self._health_check_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info(
                f"Initializing Qdrant connection pool with {self.size} connections"
            )

            # Create initial connections
            for i in range(self.size):
                try:
                    client = await self._create_client()
                    self.connections.append(client)
                    await self.available.put(client)
                    self.connection_health[client] = datetime.utcnow()
                    logger.debug(f"Created connection {i + 1}/{self.size}")
                except Exception as e:
                    logger.error(f"Failed to create connection {i + 1}: {e}")
                    # Continue with fewer connections if some fail

            if not self.connections:
                raise RuntimeError("Failed to create any Qdrant connections")

            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            self._initialized = True
            logger.info(
                f"Connection pool initialized with {len(self.connections)} connections"
            )

    async def _create_client(self) -> AsyncQdrantClient:
        """Create a new Qdrant client."""
        return AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    async def _health_check_loop(self) -> None:
        """Periodically check connection health."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def _check_all_connections(self) -> None:
        """Check health of all connections and recycle if needed."""
        now = datetime.utcnow()
        connections_to_check = []

        # Get all available connections without blocking
        while not self.available.empty():
            try:
                conn = self.available.get_nowait()
                connections_to_check.append(conn)
            except asyncio.QueueEmpty:
                break

        # Check each connection
        for conn in connections_to_check:
            try:
                # Check if connection is healthy
                collections = await conn.get_collections()
                if collections is not None:
                    # Connection is healthy
                    self.connection_health[conn] = now
                    await self.available.put(conn)
                else:
                    # Connection seems unhealthy, recycle it
                    await self._recycle_connection(conn)
            except Exception as e:
                logger.warning(f"Connection health check failed: {e}")
                await self._recycle_connection(conn)

    async def _recycle_connection(self, old_conn: AsyncQdrantClient) -> None:
        """Recycle a connection by closing it and creating a new one."""
        try:
            logger.debug("Recycling Qdrant connection")

            # Remove old connection
            if old_conn in self.connections:
                self.connections.remove(old_conn)
            if old_conn in self.connection_health:
                del self.connection_health[old_conn]

            # Try to close old connection
            with suppress(Exception):
                await old_conn.close()

            # Create new connection
            new_conn = await self._create_client()
            self.connections.append(new_conn)
            self.connection_health[new_conn] = datetime.utcnow()
            await self.available.put(new_conn)

            self.recycled_connections += 1
            logger.debug("Successfully recycled connection")

        except Exception as e:
            logger.error(f"Failed to recycle connection: {e}")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[AsyncQdrantClient, None]:
        """
        Acquire a connection from the pool.

        Usage:
            async with pool.acquire() as client:
                await client.search(...)
        """
        if not self._initialized:
            await self.initialize()

        client = None
        max_retries = 3
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                # Get connection with timeout
                client = await asyncio.wait_for(self.available.get(), timeout=5.0)

                self.total_requests += 1

                # Yield the connection
                yield client

                # Return connection to pool on success
                await self.available.put(client)
                return

            except TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Timeout acquiring connection, attempt {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(retry_delay * (2**attempt))
                else:
                    raise RuntimeError(
                        "Failed to acquire connection from pool after retries"
                    ) from None

            except UnexpectedResponse:
                # Qdrant error - recycle the connection
                self.failed_requests += 1
                if client:
                    await self._recycle_connection(client)
                raise

            except Exception:
                # Other error - recycle connection as it may be broken
                self.failed_requests += 1
                if client:
                    await self._recycle_connection(client)
                raise

    async def get_client(self) -> AsyncQdrantClient:
        """
        Get a random client from the pool for backward compatibility.
        Prefer using acquire() context manager instead.
        """
        if not self._initialized:
            await self.initialize()

        # Return a random connection for load balancing
        if self.connections:
            return random.choice(self.connections)
        else:
            raise RuntimeError("No connections available in pool")

    async def close(self) -> None:
        """Close all connections in the pool."""
        logger.info("Closing connection pool")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_check_task

        # Close all connections
        for conn in self.connections:
            try:
                await conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

        self.connections.clear()
        self.connection_health.clear()

        # Clear the queue
        while not self.available.empty():
            try:
                self.available.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._initialized = False
        logger.info("Connection pool closed")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.size,
            "active_connections": len(self.connections),
            "available_connections": self.available.qsize(),
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "recycled_connections": self.recycled_connections,
            "failure_rate": (
                self.failed_requests / self.total_requests
                if self.total_requests > 0
                else 0
            ),
        }


# Global pool instance
_pool: QdrantConnectionPool | None = None


def get_pool() -> QdrantConnectionPool:
    """Get or create the global connection pool."""
    global _pool
    if _pool is None:
        _pool = QdrantConnectionPool()
    return _pool


async def initialize_pool() -> None:
    """Initialize the global connection pool."""
    pool = get_pool()
    await pool.initialize()


async def close_pool() -> None:
    """Close the global connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
