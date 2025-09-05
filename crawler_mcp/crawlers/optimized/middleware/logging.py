from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response


class LoggingMiddleware:
    """Middleware to log HTTP requests and responses with timing information."""

    def __init__(self, app: Callable[[Request], Awaitable[Response]]):
        self.app = app
        self.logger = logging.getLogger(__name__)

    async def __call__(self, request: Request) -> Response:  # pragma: no cover - used by transport
        start = time.perf_counter()
        response = await self.app(request)
        elapsed = (time.perf_counter() - start) * 1000
        self.logger.debug(
            "HTTP %s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed
        )
        return response

