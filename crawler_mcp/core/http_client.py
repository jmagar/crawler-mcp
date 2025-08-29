"""
Unified HTTP client management for all network operations.

This module provides standardized HTTP clients with consistent configuration,
error handling, retries, and resource management across the entire codebase.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from ..config import settings
from .logging import get_logger

logger = get_logger(__name__)


class HttpClientConfig:
    """Configuration for HTTP clients with sensible defaults."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        retries: int = 3,
        follow_redirects: bool = True,
        headers: dict[str, str] | None = None,
    ):
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.retries = retries
        self.follow_redirects = follow_redirects
        self.headers = headers or {}


class AsyncHttpClient:
    """
    Reusable async HTTP client with automatic connection management.

    Features:
    - Connection pooling and reuse
    - Automatic client recreation on close
    - Configurable timeouts and limits
    - Built-in retry logic
    - Proper async context manager support
    """

    def __init__(self, config: HttpClientConfig | None = None):
        self.config = config or HttpClientConfig()
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _ensure_client_open(self) -> httpx.AsyncClient:
        """Ensure HTTP client is open and recreate if needed."""
        async with self._lock:
            if self._client is None or self._client.is_closed:
                if self._client:
                    logger.debug("Recreating closed HTTP client")

                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.config.timeout),
                    limits=httpx.Limits(
                        max_keepalive_connections=self.config.max_keepalive_connections,
                        max_connections=self.config.max_connections,
                    ),
                    follow_redirects=self.config.follow_redirects,
                    headers=self.config.headers,
                )

        return self._client

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform GET request with retry logic."""
        client = await self._ensure_client_open()

        for attempt in range(self.config.retries):
            try:
                response = await client.get(url, **kwargs)
                response.raise_for_status()
                return response

            except (httpx.TimeoutException, httpx.RequestError) as e:
                if attempt == self.config.retries - 1:
                    logger.error(
                        f"GET {url} failed after {self.config.retries} attempts: {e}"
                    )
                    raise

                logger.warning(
                    f"GET {url} attempt {attempt + 1} failed: {e}, retrying..."
                )
                await asyncio.sleep(0.5 * (2**attempt))  # Exponential backoff

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform POST request with retry logic."""
        client = await self._ensure_client_open()

        for attempt in range(self.config.retries):
            try:
                response = await client.post(url, **kwargs)
                response.raise_for_status()
                return response

            except (httpx.TimeoutException, httpx.RequestError) as e:
                if attempt == self.config.retries - 1:
                    logger.error(
                        f"POST {url} failed after {self.config.retries} attempts: {e}"
                    )
                    raise

                logger.warning(
                    f"POST {url} attempt {attempt + 1} failed: {e}, retrying..."
                )
                await asyncio.sleep(0.5 * (2**attempt))

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Perform any HTTP request with retry logic."""
        client = await self._ensure_client_open()

        for attempt in range(self.config.retries):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except (httpx.TimeoutException, httpx.RequestError) as e:
                if attempt == self.config.retries - 1:
                    logger.error(
                        f"{method} {url} failed after {self.config.retries} attempts: {e}"
                    )
                    raise

                logger.warning(
                    f"{method} {url} attempt {attempt + 1} failed: {e}, retrying..."
                )
                await asyncio.sleep(0.5 * (2**attempt))

    async def health_check(self, url: str) -> bool:
        """Perform a simple health check request."""
        try:
            response = await self.get(url)
            return bool(200 <= response.status_code < 300)
        except Exception as e:
            logger.debug(f"Health check failed for {url}: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncHttpClient":
        """Async context manager entry."""
        await self._ensure_client_open()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


class HttpClientFactory:
    """Factory for creating configured HTTP clients."""

    @staticmethod
    def create_default() -> AsyncHttpClient:
        """Create HTTP client with default configuration."""
        return AsyncHttpClient()

    @staticmethod
    def create_tei_client() -> AsyncHttpClient:
        """Create HTTP client configured for TEI service."""
        config = HttpClientConfig(
            timeout=settings.tei_timeout,
            max_connections=settings.tei_max_concurrent_requests * 2,
            max_keepalive_connections=settings.tei_max_concurrent_requests,
            retries=3,
        )
        return AsyncHttpClient(config)

    @staticmethod
    def create_github_client(token: str | None = None) -> AsyncHttpClient:
        """Create HTTP client configured for GitHub API."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"

        config = HttpClientConfig(
            timeout=30.0,
            max_connections=50,
            max_keepalive_connections=10,
            retries=3,
            headers=headers,
        )
        return AsyncHttpClient(config)

    @staticmethod
    def create_web_scraping_client() -> AsyncHttpClient:
        """Create HTTP client configured for web scraping."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        config = HttpClientConfig(
            timeout=45.0,
            max_connections=20,
            max_keepalive_connections=5,
            retries=2,
            headers=headers,
        )
        return AsyncHttpClient(config)


@contextlib.asynccontextmanager
async def get_http_client(
    config: HttpClientConfig | None = None,
) -> AsyncGenerator["AsyncHttpClient", None]:
    """
    Async context manager for getting a configured HTTP client.

    Usage:
        async with get_http_client() as client:
            response = await client.get("https://example.com")
    """
    client = AsyncHttpClient(config)
    try:
        yield client
    finally:
        await client.close()


@contextlib.asynccontextmanager
async def get_tei_client() -> AsyncGenerator["AsyncHttpClient", None]:
    """Async context manager for TEI service HTTP client."""
    client = HttpClientFactory.create_tei_client()
    try:
        yield client
    finally:
        await client.close()


@contextlib.asynccontextmanager
async def get_github_client(
    token: str | None = None,
) -> AsyncGenerator["AsyncHttpClient", None]:
    """Async context manager for GitHub API HTTP client."""
    client = HttpClientFactory.create_github_client(token)
    try:
        yield client
    finally:
        await client.close()


@contextlib.asynccontextmanager
async def get_web_scraping_client() -> AsyncGenerator["AsyncHttpClient", None]:
    """Async context manager for web scraping HTTP client."""
    client = HttpClientFactory.create_web_scraping_client()
    try:
        yield client
    finally:
        await client.close()


# Backwards compatibility - simple function for basic usage
async def simple_get(url: str, timeout: float = 30.0) -> str:
    """
    Simple GET request returning text content.

    Replacement for basic aiohttp usage patterns.
    """
    try:
        async with get_http_client() as client:
            response = await client.get(url)
            content_bytes = await response.aread()
            return str(content_bytes.decode("utf-8"))
    except Exception as e:
        logger.warning(f"Simple GET request to {url} failed: {e}")
        return ""
