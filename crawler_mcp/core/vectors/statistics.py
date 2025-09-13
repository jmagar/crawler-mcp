"""
Statistics and analytics operations for Qdrant vector database.
"""

import logging
from typing import Any
from urllib.parse import urlparse

from qdrant_client import AsyncQdrantClient

from .base import BaseVectorService

logger = logging.getLogger(__name__)


class StatisticsCollector(BaseVectorService):
    """
    Collects and analyzes vector database statistics and source information.
    """

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        """
        Initialize the statistics collector.

        Args:
            client: Optional shared Qdrant client instance
        """
        super().__init__(client)

    async def get_sources_stats(
        self, lightweight: bool = True, max_sources: int = 20
    ) -> dict[str, Any]:
        """
        Get statistics about sources in the vector database.

        Args:
            lightweight: If True, returns only summary statistics without detailed source counts
            max_sources: Maximum number of top sources to include in detailed stats

        Returns:
            Dictionary with source statistics
        """
        # Ensure collection exists
        from .collections import CollectionManager

        collection_manager = CollectionManager(self.client)
        await collection_manager.ensure_collection_exists()

        try:
            # Get collection info for basic stats
            client = await self._get_client()
            collection_info = await client.get_collection(self.collection_name)
            total_points = collection_info.points_count or 0

            if lightweight:
                # Return lightweight stats without full scan
                return {
                    "total_documents": total_points,
                    "unique_sources": "unknown (lightweight mode)",
                    "source_counts": "use detailed mode for source breakdown",
                    "total_content_length": "unknown (lightweight mode)",
                    "average_chunk_size": "unknown (lightweight mode)",
                    "mode": "lightweight",
                }

            # Full scan for detailed statistics
            stats: dict[str, Any] = {
                "total_documents": 0,
                "unique_sources": set(),
                "source_counts": {},
                "total_content_length": 0,
                "average_chunk_size": 0.0,
            }

            offset = None
            limit = 1000

            while True:
                result = await client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points = result[0]
                next_offset = result[1]

                if not points:
                    break

                for point in points:
                    payload = getattr(point, "payload", None)
                    if payload is None:
                        continue
                    source_url = payload.get("source_url", "unknown")
                    char_count = payload.get("char_count", 0)

                    stats["total_documents"] += 1
                    stats["unique_sources"].add(source_url)
                    stats["source_counts"][source_url] = (
                        stats["source_counts"].get(source_url, 0) + 1
                    )
                    stats["total_content_length"] += char_count

                if next_offset is None:
                    break
                offset = next_offset

            # Calculate average chunk size
            if stats["total_documents"] > 0:
                stats["average_chunk_size"] = (
                    stats["total_content_length"] / stats["total_documents"]
                )

            # Limit source counts to top N sources
            if len(stats["source_counts"]) > max_sources:
                total_sources = len(stats["source_counts"])
                top_sources = dict(
                    sorted(
                        stats["source_counts"].items(), key=lambda x: x[1], reverse=True
                    )[:max_sources]
                )
                top_sources["...and_more"] = (
                    f"{total_sources - max_sources} more sources"
                )
                stats["source_counts"] = top_sources

            # Convert set to count
            stats["unique_sources"] = len(stats["unique_sources"])
            stats["mode"] = "detailed"

            return stats

        except Exception as e:
            logger.error(f"Error getting source statistics: {e}")
            return {}

    async def get_unique_sources(
        self,
        domains: list[str] | None = None,
        search_term: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get unique sources from the vector database with filtering and pagination.

        Args:
            domains: Filter by domains (e.g., ["github.com", "example.com"])
            search_term: Search term to filter titles and URLs
            limit: Maximum number of sources to return
            offset: Offset for pagination

        Returns:
            Dictionary with sources and metadata
        """
        # Ensure collection exists
        from .collections import CollectionManager

        collection_manager = CollectionManager(self.client)
        await collection_manager.ensure_collection_exists()

        try:
            # Collect source information
            sources_data: dict[str, dict[str, Any]] = {}
            total_sources = 0

            # Scroll through all points to collect source information
            scroll_offset = None
            scroll_limit = 1000

            while True:
                client = await self._get_client()
                result = await client.scroll(
                    collection_name=self.collection_name,
                    limit=scroll_limit,
                    offset=scroll_offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points = result[0]
                next_offset = result[1]

                if not points:
                    break

                for point in points:
                    payload = getattr(point, "payload", None)
                    if payload is None:
                        continue

                    source_url = payload.get("source_url", "")
                    source_title = payload.get("source_title", "")
                    char_count = payload.get("char_count", 0)
                    word_count = payload.get("word_count", 0)
                    timestamp = payload.get("timestamp", "")

                    if not source_url:
                        continue

                    # Apply domain filtering
                    if domains and not await self._matches_domains(source_url, domains):
                        continue

                    # Apply search term filtering
                    if search_term and not await self._matches_search_term(
                        source_url, source_title, search_term
                    ):
                        continue

                    # Aggregate source data
                    if source_url not in sources_data:
                        sources_data[source_url] = {
                            "url": source_url,
                            "title": source_title,
                            "chunk_count": 0,
                            "total_content_length": 0,
                            "total_word_count": 0,
                            "last_crawled": timestamp,
                            "source_type": "webpage",  # Default, could be enhanced
                            "status": "active",
                        }

                    # Update aggregated stats
                    source_data = sources_data[source_url]
                    source_data["chunk_count"] += 1
                    source_data["total_content_length"] += char_count
                    source_data["total_word_count"] += word_count

                    # Keep the most recent timestamp
                    if timestamp and (
                        not source_data["last_crawled"]
                        or timestamp > source_data["last_crawled"]
                    ):
                        source_data["last_crawled"] = timestamp

                if next_offset is None:
                    break
                scroll_offset = next_offset

            # Convert to list and apply pagination
            all_sources = list(sources_data.values())
            total_sources = len(all_sources)

            # Sort by URL for consistent ordering
            all_sources.sort(key=lambda x: x["url"])

            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            paginated_sources = all_sources[start_idx:end_idx]

            return {
                "sources": paginated_sources,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total_sources,
                    "returned": len(paginated_sources),
                },
                "filters_applied": {
                    "domains": domains,
                    "search_term": search_term,
                },
            }

        except Exception as e:
            logger.error(f"Error getting unique sources: {e}")
            return {
                "sources": [],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": 0,
                    "returned": 0,
                },
                "filters_applied": {"domains": domains, "search_term": search_term},
                "error": str(e),
            }

    async def get_client(self) -> AsyncQdrantClient:
        """Public accessor for Qdrant client to keep encapsulation."""
        return await self._get_client()

    async def _matches_domains(self, source_url: str, domains: list[str]) -> bool:
        """
        Check if source URL matches any of the specified domains.

        Args:
            source_url: URL to check
            domains: List of domains to match against

        Returns:
            True if URL matches any domain
        """
        try:
            parsed_url = urlparse(source_url)
            host = parsed_url.netloc.lower()
            domains_lc = {d.lower() for d in domains}
            return any(host == d or host.endswith(f".{d}") for d in domains_lc)
        except Exception:
            return False

    async def _matches_search_term(
        self, source_url: str, source_title: str, search_term: str
    ) -> bool:
        """
        Check if source URL or title matches the search term.

        Args:
            source_url: URL to search in
            source_title: Title to search in
            search_term: Term to search for

        Returns:
            True if search term is found
        """
        search_lower = search_term.lower()
        return (
            search_lower in source_url.lower()
            or search_lower in (source_title or "").lower()
        )

    async def get_collection_health(self) -> dict[str, Any]:
        """
        Get collection health metrics.

        Returns:
            Dictionary with health information
        """
        try:
            from .collections import CollectionManager

            collection_manager = CollectionManager(self.client)

            # Get collection info
            collection_info = await collection_manager.get_collection_info()

            # Calculate health metrics
            health_score = 100.0
            issues = []

            # Check if collection has data
            if collection_info.get("points_count", 0) == 0:
                health_score -= 30
                issues.append("No data points in collection")

            # Check indexing status
            vectors_count = collection_info.get("vectors_count", 0)
            indexed_count = collection_info.get("indexed_vectors_count", 0)

            if vectors_count > 0:
                indexing_ratio = indexed_count / vectors_count
                if indexing_ratio < 0.9:
                    health_score -= 20
                    issues.append(f"Low indexing ratio: {indexing_ratio:.1%}")

            return {
                "health_score": health_score,
                "status": "healthy"
                if health_score >= 80
                else "degraded"
                if health_score >= 60
                else "unhealthy",
                "issues": issues,
                "collection_info": collection_info,
            }

        except Exception as e:
            logger.error(f"Error getting collection health: {e}")
            return {
                "health_score": 0.0,
                "status": "error",
                "issues": [f"Failed to get health metrics: {e}"],
                "collection_info": {},
            }

    async def analyze_content_distribution(self) -> dict[str, Any]:
        """
        Analyze content distribution across sources.

        Returns:
            Dictionary with distribution analysis
        """
        try:
            # Get basic stats (lightweight mode for better performance)
            stats = await self.get_sources_stats(lightweight=True)

            # Additional analysis could be added here
            # For now, return basic distribution info
            return {
                "total_sources": stats.get("unique_sources", 0),
                "total_documents": stats.get("total_documents", 0),
                "average_docs_per_source": (
                    stats.get("total_documents", 0)
                    / max(1, stats.get("unique_sources", 1))
                ),
                "average_chunk_size": stats.get("average_chunk_size", 0.0),
            }

        except Exception as e:
            logger.error(f"Error analyzing content distribution: {e}")
            return {}

    async def get_embedding_quality_metrics(self) -> dict[str, Any]:
        """
        Get metrics about embedding quality and performance.

        Returns:
            Dictionary with embedding quality metrics
        """
        # Placeholder for future embedding quality analysis
        # This could include metrics like:
        # - Average vector magnitude
        # - Distribution of similarity scores
        # - Clustering analysis
        logger.info("Embedding quality metrics not yet implemented")
        return {
            "status": "not_implemented",
            "message": "Embedding quality analysis is planned for future release",
        }
