"""
Vector similarity search operations for Qdrant vector database.
"""

import logging
from datetime import datetime
from typing import Any

from fastmcp.exceptions import ToolError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    Range,
    SearchParams,
)

from crawler_mcp.settings import get_settings

from ...models.rag import DocumentChunk, SearchMatch
from .base import BaseVectorService, _parse_timestamp

settings = get_settings()

logger = logging.getLogger(__name__)


class SearchEngine(BaseVectorService):
    """
    Handles vector similarity search operations and query optimization.
    """

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        """
        Initialize the search engine.

        Args:
            client: Optional shared Qdrant client instance
        """
        super().__init__(client)

    async def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        source_filter: list[str] | None = None,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> list[SearchMatch]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            source_filter: Filter by source URLs (optional)
            date_range: Filter by date range as ISO strings (optional)

        Returns:
            List of search matches with similarity scores
        """
        # Ensure collection exists
        from .collections import CollectionManager

        collection_manager = CollectionManager(self.client)
        await collection_manager.ensure_collection_exists()

        try:
            # Build filters
            search_filter = await self._build_search_filters(source_filter, date_range)

            # Optimize search parameters based on query complexity
            search_params = await self._optimize_search_params(limit)

            # Validate vector dimension
            if len(query_vector) != self.vector_size:
                raise ToolError(
                    f"Query vector size {len(query_vector)} != expected {self.vector_size}"
                )

            # Perform search using modern query_points API
            client = await self._get_client()
            query_response = await client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,  # Don't return vectors to save bandwidth
                search_params=search_params,
            )

            # Extract and convert results to SearchMatch objects
            matches = await self._process_search_results(query_response)

            logger.debug(f"Found {len(matches)} similar documents")
            return matches

        except Exception as e:
            logger.error(f"Error searching for similar documents: {e}")
            raise ToolError(f"Vector search failed: {e!s}") from e

    async def _build_search_filters(
        self,
        source_filter: list[str] | None = None,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> Filter | None:
        """
        Build search filters for the query.

        Args:
            source_filter: Filter by source URLs
            date_range: Filter by date range as datetimes

        Returns:
            Qdrant Filter object or None if no filters
        """
        filter_conditions: list[FieldCondition] = []
        should_conditions: list[FieldCondition] = []

        # OR-match both "source_url" and "url" keys for source filtering
        if source_filter:
            any_match = MatchAny(any=source_filter)
            should_conditions.extend(
                [
                    FieldCondition(key="source_url", match=any_match),
                    FieldCondition(key="url", match=any_match),
                ]
            )

        # Add date range filtering using a single Range
        if date_range:
            start, end = date_range
            range_kwargs: dict[str, str] = {}
            if start is not None:
                range_kwargs["gte"] = _parse_timestamp(start).isoformat()
            if end is not None:
                range_kwargs["lte"] = _parse_timestamp(end).isoformat()
            if range_kwargs:
                filter_conditions.append(
                    FieldCondition(key="timestamp", range=Range(**range_kwargs))
                )

        # Return filter if we have conditions
        if filter_conditions or should_conditions:
            return Filter(
                must=list(filter_conditions) if filter_conditions else None,
                should=should_conditions or None,
            )
        return None

    async def _optimize_search_params(self, limit: int) -> SearchParams:
        """
        Optimize search parameters based on query complexity.

        Args:
            limit: Number of results requested

        Returns:
            Optimized SearchParams object
        """
        # Dynamic ef parameter based on query complexity
        # Higher ef for better accuracy when needed
        ef_value = min(256, max(64, limit * 4))  # 4x limit, capped at 256

        return SearchParams(
            hnsw_ef=ef_value,
            exact=settings.qdrant_search_exact,
        )

    async def _process_search_results(self, query_response: Any) -> list[SearchMatch]:
        """
        Process search results and convert to SearchMatch objects.

        Args:
            query_response: Raw response from Qdrant query

        Returns:
            List of SearchMatch objects
        """
        # Extract results - handle different return types from query_points
        results: list[Any] = []
        if hasattr(query_response, "points"):
            results = query_response.points
        elif isinstance(query_response, (tuple, list)) and len(query_response) > 0:  # noqa: UP038
            results = query_response[0]
        else:
            # Cast to list for consistent handling
            if hasattr(query_response, "__iter__"):
                results = list(query_response)
            else:
                results = [query_response]

        # Convert results to SearchMatch objects
        matches = []
        for result in results:
            # Safely access payload with null checks
            payload = getattr(result, "payload", None)
            if payload is None:
                continue

            # Safely access result attributes
            result_id = getattr(result, "id", "unknown")
            result_score = getattr(result, "score", 0.0)

            # Reconstruct document chunk
            document = DocumentChunk(
                id=str(result_id),
                content=payload.get("content", ""),
                source_url=payload.get("source_url", ""),
                source_title=payload.get("source_title"),
                chunk_index=payload.get("chunk_index", 0),
                word_count=payload.get("word_count", 0),
                char_count=payload.get("char_count", 0),
                timestamp=_parse_timestamp(payload.get("timestamp", "")),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k
                    not in [
                        "content",
                        "source_url",
                        "source_title",
                        "chunk_index",
                        "word_count",
                        "char_count",
                        "timestamp",
                    ]
                },
            )

            # Create search match
            match = SearchMatch(document=document, score=float(result_score))
            matches.append(match)

        return matches

    async def search_with_reranking(
        self, query_vector: list[float], rerank_threshold: float = 0.8, **kwargs: Any
    ) -> list[SearchMatch]:
        """
        Perform search with optional result reranking.

        Args:
            query_vector: Query embedding vector
            rerank_threshold: Score threshold for reranking
            **kwargs: Additional arguments for search_similar

        Returns:
            List of reranked search matches
        """
        # Perform initial search
        results = await self.search_similar(query_vector, **kwargs)

        # Apply rerank threshold filtering
        if rerank_threshold is not None:
            results = [m for m in results if m.score >= float(rerank_threshold)]

        # Simple reranking based on content length and score combination
        # In a production system, this could use more sophisticated reranking
        if results and len(results) > 1:
            # Boost scores for results with balanced content length
            for match in results:
                content_length = len(match.document.content)
                # Optimal content length is around 500-1500 characters
                if 500 <= content_length <= 1500:
                    match.score *= 1.1  # Small boost for well-sized content

            # Sort by adjusted scores
            results.sort(key=lambda x: x.score, reverse=True)

        return results
