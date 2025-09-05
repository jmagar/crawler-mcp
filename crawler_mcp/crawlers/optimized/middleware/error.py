from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastmcp.exceptions import ToolError


class ErrorHandlingMiddleware:
    """Middleware to handle errors in FastMCP operations with proper logging"""

    def __init__(self, app: Callable[[Request], Awaitable[Response]]):
        self.app = app
        self.logger = logging.getLogger(__name__)

    async def __call__(self, request: Request) -> Response:  # pragma: no cover - used by transport
        try:
            return await self.app(request)
        except ToolError:
            raise
        except Exception as e:
            self.logger.exception("Unhandled error in request: %s", e)
            return Response("Internal Server Error", status_code=500)

