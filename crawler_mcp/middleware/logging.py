"""
Logging middleware for FastMCP operations.
"""

import logging
import time
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware:
    """
    Middleware to log HTTP requests and responses with timing information.
    """

    def __init__(self, app: Callable[[Request], Awaitable[Response]]):
        self.app = app

    async def __call__(self, request: Request) -> Response:
        """
        Log incoming requests and responses with timing.
        """
        start_time = time.time()

        # Extract useful request info
        method = request.method
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "Unknown")

        # Log incoming request
        logger.info(
            f"Incoming {method} {path} from {client_ip} {'?' + query if query else ''}"
        )

        # Log user agent for debugging
        logger.debug(f"User-Agent: {user_agent}")

        try:
            # Process the request
            response = await self.app(request)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Log successful response
            logger.info(
                f"Completed {method} {path} -> {response.status_code} "
                f"in {processing_time:.3f}s"
            )

            # Add timing header to response
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"

            return response

        except Exception as e:
            # Calculate processing time for failed requests
            processing_time = time.time() - start_time

            # Log error response
            logger.error(f"Failed {method} {path} after {processing_time:.3f}s: {e!s}")

            # Re-raise the exception to be handled by error middleware
            raise

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request headers.
        """
        # Check for forwarded headers first (for proxies/load balancers)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, take the first one
            return str(forwarded_for.split(",")[0].strip())

        # Check other common headers
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return str(real_ip.strip())

        # Fall back to direct client IP
        if hasattr(request, "client") and request.client:
            return str(request.client.host)

        return "unknown"
