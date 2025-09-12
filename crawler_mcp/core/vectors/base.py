"""
Base class for vector service modules providing shared functionality.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from dateutil import parser as date_parser
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance

from crawler_mcp.settings import get_settings

settings = get_settings()
from ..connection_pool import get_pool

logger = logging.getLogger(__name__)


def _parse_timestamp(timestamp_value: Any) -> datetime:
    """
    Parse timestamp from various formats to timezone-aware datetime in UTC.
    Handles ISO strings, datetime objects, and empty values.
    All returned datetimes are guaranteed to be timezone-aware in UTC.
    """
    if isinstance(timestamp_value, datetime):
        # If datetime has no timezone, attach UTC; otherwise leave as-is
        if timestamp_value.tzinfo is None:
            return timestamp_value.replace(tzinfo=UTC)
        return timestamp_value

    if isinstance(timestamp_value, str) and timestamp_value:
        try:
            parsed_dt = date_parser.parse(timestamp_value)
            # If parsed datetime has no timezone, set it to UTC
            if parsed_dt.tzinfo is None:
                return parsed_dt.replace(tzinfo=UTC)
            return parsed_dt
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse timestamp: {timestamp_value}")

    # Default to current time in UTC for invalid/empty timestamps
    return datetime.now(UTC)


class BaseVectorService:
    """
    Base class for vector service modules providing shared client management
    and common functionality.
    """

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        """
        Initialize the base vector service.

        Args:
            client: Optional shared Qdrant client instance
        """
        self.pool = get_pool()
        self._owned_client = client is None

        if client is not None:
            self.client = client
        else:
            # Will get from pool when needed
            self.client = None

        self.collection_name = settings.qdrant_collection
        self.vector_size = settings.qdrant_vector_size

        # Map distance strings to Qdrant Distance enum
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        self.distance = distance_map.get(
            settings.qdrant_distance.lower(), Distance.COSINE
        )

    async def _get_client(self) -> AsyncQdrantClient:
        """Get a client from the pool or use existing."""
        if self.client is not None:
            return self.client
        # Get from pool for operations
        return await self.pool.get_client()

    async def _recreate_client(self) -> None:
        """Recreate the Qdrant client (compatibility method)."""
        logger.debug("Client recreation requested - using pool for new client")
        # Pool handles recycling automatically
        self.client = None

    async def _handle_client_error(self, e: Exception) -> bool:
        """
        Handle client connection errors with automatic recreation.

        Args:
            e: The exception that occurred

        Returns:
            True if client was recreated, False if error should be re-raised
        """
        if "client has been closed" in str(e):
            logger.debug("Client closed error detected, recreating client...")
            await self._recreate_client()
            return True
        return False

    async def close(self) -> None:
        """Close the Qdrant client."""
        # Only close if we own the client and it's not from pool
        if self.client and self._owned_client:
            try:
                await self.client.close()
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {e}")
        self.client = None
