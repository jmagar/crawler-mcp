"""
Modular vector service for Qdrant operations.

This module provides a unified interface to vector database operations
while maintaining backward compatibility with the original VectorService.
"""

import logging
from datetime import datetime
from typing import Any

from ...models.rag import DocumentChunk, SearchMatch
from .base import BaseVectorService
from .collections import CollectionManager
from .operations import DocumentOperations
from .search import SearchEngine
from .statistics import StatisticsCollector

logger = logging.getLogger(__name__)


class VectorService(BaseVectorService):
    """
    Unified vector service using modular components.

    Provides the same interface as the original VectorService while
    delegating operations to specialized modules for better organization.
    """

    def __init__(self) -> None:
        """Initialize the modular vector service."""
        super().__init__()

        # Initialize all modules with shared client
        self.collections = CollectionManager(self.client)
        self.operations = DocumentOperations(self.client)
        self.search = SearchEngine(self.client)
        self.statistics = StatisticsCollector(self.client)

    async def __aenter__(self) -> "VectorService":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type, exc_val: Exception, exc_tb: object
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the Qdrant client."""
        await super().close()

    # Backward compatibility methods - delegate to appropriate modules

    async def health_check(self) -> bool:
        """Check if Qdrant service is healthy and responsive."""
        return await self.collections.health_check()

    async def ensure_collection(self) -> bool:
        """Ensure the collection exists, create if it doesn't."""
        return await self.collections.ensure_collection_exists()

    async def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        return await self.collections.get_collection_info()

    async def upsert_documents(
        self, documents: list[DocumentChunk], batch_size: int = 100
    ) -> int:
        """Upsert document chunks into the vector database."""
        return await self.operations.upsert_documents(documents, batch_size)

    async def get_document_by_id(self, document_id: str) -> DocumentChunk | None:
        """Retrieve a specific document by ID."""
        return await self.operations.get_document_by_id(document_id)

    async def delete_documents_by_source(self, source_url: str) -> int:
        """Delete all documents from a specific source."""
        return await self.operations.delete_documents_by_source(source_url)

    async def get_chunks_by_source(self, source_url: str) -> list[dict[str, Any]]:
        """Get all existing chunks for a source URL."""
        return await self.operations.get_chunks_by_source(source_url)

    async def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
        """Delete specific chunks by their IDs."""
        return await self.operations.delete_chunks_by_ids(chunk_ids)

    async def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        source_filter: list[str] | None = None,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> list[SearchMatch]:
        """Search for similar documents using vector similarity."""
        return await self.search.search_similar(
            query_vector, limit, score_threshold, source_filter, date_range
        )

    async def get_sources_stats(
        self, lightweight: bool = True, max_sources: int = 20
    ) -> dict[str, Any]:
        """Get statistics about sources in the vector database."""
        return await self.statistics.get_sources_stats(lightweight, max_sources)

    async def get_unique_sources(
        self,
        domains: list[str] | None = None,
        search_term: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get unique sources from the vector database with filtering and pagination."""
        return await self.statistics.get_unique_sources(
            domains, search_term, limit, offset
        )

    # Additional methods for accessing modules directly

    def get_collections_manager(self) -> CollectionManager:
        """Get the collections manager for advanced collection operations."""
        return self.collections

    def get_operations_manager(self) -> DocumentOperations:
        """Get the operations manager for advanced document operations."""
        return self.operations

    def get_search_engine(self) -> SearchEngine:
        """Get the search engine for advanced search operations."""
        return self.search

    def get_statistics_collector(self) -> StatisticsCollector:
        """Get the statistics collector for advanced analytics."""
        return self.statistics

    # Recreate client method for backward compatibility
    async def _recreate_client(self) -> None:
        """Recreate the Qdrant client and update all modules."""
        await super()._recreate_client()

        # Update client reference in all modules
        self.collections.client = self.client
        self.operations.client = self.client
        self.search.client = self.client
        self.statistics.client = self.client


# Export all classes for direct use
__all__ = [
    "BaseVectorService",
    "CollectionManager",
    "DocumentOperations",
    "SearchEngine",
    "StatisticsCollector",
    "VectorService",
]


# Convenience function to create a VectorService instance
def create_vector_service() -> VectorService:
    """
    Create a new VectorService instance.

    Returns:
        Configured VectorService instance
    """
    return VectorService()


# Legacy import compatibility - allows imports like:
# from crawler_mcp.core.vectors import VectorService
# This maintains backward compatibility with existing code
VectorServiceLegacy = VectorService
