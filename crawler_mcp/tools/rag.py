"""
FastMCP tools for RAG operations.
"""

import logging
from datetime import datetime
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from ..core import RagService, VectorService
from ..middleware.progress import progress_middleware
from ..models.rag import RagQuery

logger = logging.getLogger(__name__)


def register_rag_tools(mcp: FastMCP) -> None:
    """Register all RAG tools with the FastMCP server."""

    @mcp.tool
    async def rag_query(
        ctx: Context,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        source_filters: list[str] | None = None,
        rerank: bool = True,
        include_content: bool = True,
    ) -> dict[str, Any]:
        """
        Perform semantic search using RAG to find relevant documents.

        Args:
            query: Search query text
            limit: Maximum number of results to return (1-100)
            min_score: Minimum similarity score threshold (0.0-1.0)
            source_filters: List of source URLs to filter by (optional)
            rerank: Whether to apply re-ranking to improve results
            include_content: Whether to include full content in results

        Returns:
            Dictionary with search results and metadata
        """
        await ctx.info(f"Performing RAG query: '{query}' (limit: {limit})")

        # Validate parameters
        if limit > 100:
            raise ToolError("limit cannot exceed 100")
        if not 0.0 <= min_score <= 1.0:
            raise ToolError("min_score must be between 0.0 and 1.0")
        if not query.strip():
            raise ToolError("query cannot be empty")

        # Create progress tracker
        progress_tracker = progress_middleware.create_tracker(f"rag_{hash(query)}")

        try:
            await ctx.report_progress(progress=1, total=4)

            # Create RAG query object
            rag_query_obj = RagQuery(
                query=query.strip(),
                limit=limit,
                min_score=min_score,
                source_filters=source_filters,
                include_content=include_content,
                rerank=rerank,
            )

            await ctx.report_progress(progress=2, total=4)

            # Perform RAG query
            async with RagService() as rag_service:
                await ctx.report_progress(progress=3, total=4)

                rag_result = await rag_service.query(rag_query_obj, rerank=rerank)

            await ctx.report_progress(progress=4, total=4)

            # Prepare response
            result: dict[str, Any] = {
                "query": rag_result.query,
                "total_matches": rag_result.total_matches,
                "matches": [],
                "performance": {
                    "total_time": rag_result.processing_time,
                    "embedding_time": rag_result.embedding_time,
                    "search_time": rag_result.search_time,
                    "rerank_time": rag_result.rerank_time,
                },
                "quality_metrics": {
                    "average_score": rag_result.average_score,
                    "best_match_score": rag_result.best_match_score,
                    "high_confidence_matches": rag_result.has_high_confidence_matches,
                },
                "timestamp": rag_result.timestamp.isoformat(),
            }

            # Process matches
            for match in rag_result.matches:
                match_data: dict[str, Any] = {
                    "score": match.score,
                    "relevance": match.relevance,
                    "document": {
                        "id": match.document.id,
                        "source_url": match.document.source_url,
                        "source_title": match.document.source_title,
                        "chunk_index": match.document.chunk_index,
                        "word_count": match.document.word_count,
                        "timestamp": match.document.timestamp.isoformat(),
                    },
                }

                # Include content if requested
                if include_content:
                    match_data["document"]["content"] = match.document.content
                    if match.highlighted_content:
                        match_data["highlighted_content"] = match.highlighted_content

                # Include metadata if available
                if match.document.metadata:
                    match_data["document"]["metadata"] = match.document.metadata

                result["matches"].append(match_data)

            await ctx.info(
                f"RAG query completed: {rag_result.total_matches} matches found in {rag_result.processing_time:.3f}s "
                f"(avg score: {rag_result.average_score:.3f})"
            )

            return result

        except Exception as e:
            error_msg = f"RAG query failed: {e!s}"
            await ctx.info(error_msg)
            raise ToolError(error_msg) from e

        finally:
            progress_middleware.remove_tracker(progress_tracker.operation_id)

    @mcp.tool
    async def list_sources(
        ctx: Context,
        source_types: list[str] | None = None,
        domains: list[str] | None = None,
        statuses: list[str] | None = None,
        search_term: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List and filter crawled sources with their metadata.

        Args:
            source_types: Filter by source types ("webpage", "repository", "directory")
            domains: Filter by domains (e.g., ["github.com", "example.com"])
            statuses: Filter by status ("active", "inactive", "error")
            search_term: Search term to filter titles and URLs
            limit: Maximum number of sources to return (1-200)
            offset: Offset for pagination

        Returns:
            Dictionary with filtered sources and metadata
        """
        await ctx.info(f"Listing sources (limit: {limit}, offset: {offset})")

        # Validate parameters
        if limit > 200:
            raise ToolError("limit cannot exceed 200")
        if offset < 0:
            raise ToolError("offset cannot be negative")

        try:
            # Get sources directly from VectorService
            async with VectorService() as vector_service:
                # Get unique sources with filtering
                sources_response = await vector_service.get_unique_sources(
                    domains=domains,
                    search_term=search_term,
                    limit=limit,
                    offset=offset,
                )

                # Get vector database statistics (lightweight to reduce token usage)
                vector_stats = await vector_service.get_sources_stats(lightweight=True)

            # Prepare response with simplified source data
            result: dict[str, Any] = {
                "sources": [],
                "pagination": sources_response["pagination"],
                "statistics": {
                    "vector_database": vector_stats,
                    "source_registry": {
                        "registered_sources": 0,  # No longer maintained
                        "sources_by_type": {},
                        "sources_by_status": {},
                        "stale_sources": 0,
                    },
                },
                "filters_applied": {
                    "source_types": source_types,  # Note: source_types filtering not yet implemented
                    "domains": domains,
                    "statuses": statuses,  # Note: statuses filtering not yet implemented
                    "search_term": search_term,
                },
            }

            # Process sources with simplified data structure
            for source in sources_response["sources"]:
                from urllib.parse import urlparse

                parsed_url = urlparse(source["url"])

                source_data = {
                    "id": f"src_{hash(source['url']) & 0x7FFFFFFF:08x}",  # Generate consistent ID
                    "url": source["url"],
                    "title": source["title"],
                    "source_type": source["source_type"],
                    "status": source["status"],
                    "chunk_count": source["chunk_count"],
                    "total_content_length": source["total_content_length"],
                    "average_chunk_size": (
                        source["total_content_length"] / source["chunk_count"]
                        if source["chunk_count"] > 0
                        else 0
                    ),
                    "created_at": source[
                        "last_crawled"
                    ],  # Use last_crawled as created_at
                    "updated_at": source[
                        "last_crawled"
                    ],  # Use last_crawled as updated_at
                    "last_crawled": source["last_crawled"],
                    "is_stale": False,  # Could be enhanced with timestamp logic
                    "metadata": {
                        "domain": parsed_url.netloc,
                        "word_count": source["total_word_count"],
                        "character_count": source["total_content_length"],
                        "link_count": 0,  # Not available from Qdrant
                        "image_count": 0,  # Not available from Qdrant
                        "language": None,  # Not available from Qdrant
                        "content_type": "text/html",  # Default
                        "tags": [],
                        "categories": [],
                    },
                }

                result["sources"].append(source_data)

            await ctx.info(f"Found {len(result['sources'])} sources matching criteria")

            return result

        except Exception as e:
            error_msg = f"Failed to list sources: {e!s}"
            await ctx.info(error_msg)
            raise ToolError(error_msg) from e

    @mcp.tool
    async def get_rag_stats(ctx: Context) -> dict[str, Any]:
        """
        Get comprehensive statistics about the RAG system.

        Returns:
            Dictionary with RAG system statistics and health information
        """
        await ctx.info("Retrieving RAG system statistics")

        try:
            async with RagService() as rag_service:
                stats = await rag_service.get_stats()

            async with VectorService() as vector_service:
                vector_stats = await vector_service.get_sources_stats(lightweight=True)

            # Combine statistics with simplified source management
            result: dict[str, Any] = {
                "rag_system": stats,
                "source_management": {
                    "vector_database": vector_stats,
                    "source_registry": {
                        "registered_sources": 0,  # No longer maintained
                        "sources_by_type": {},
                        "sources_by_status": {},
                        "stale_sources": 0,
                    },
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add health summary
            health = stats.get("health", {})
            all_healthy = all(health.values()) if health else False

            result["health_summary"] = {
                "all_services_healthy": all_healthy,
                "services": health,
            }

            # Add quick facts - use actual data from vector database
            collection_info = stats.get("collection", {})
            result["quick_facts"] = {
                "total_documents": collection_info.get("points_count", 0),
                "total_vectors": collection_info.get("vectors_count", 0),
                "total_sources": vector_stats.get(
                    "unique_sources", 0
                ),  # Use actual count from Qdrant
                "vector_dimension": collection_info.get("config", {}).get(
                    "vector_size", 0
                ),
                "collection_status": collection_info.get("status", "unknown"),
            }

            await ctx.info(
                f"RAG stats retrieved: {result['quick_facts']['total_documents']} documents, {result['quick_facts']['total_sources']} sources"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to get RAG statistics: {e!s}"
            await ctx.info(error_msg)
            raise ToolError(error_msg) from e

    @mcp.tool
    async def delete_source(
        ctx: Context, source_url: str, confirm: bool = False
    ) -> dict[str, Any]:
        """
        Delete a source and all its associated documents from the RAG system.

        Args:
            source_url: URL of the source to delete
            confirm: Confirmation flag (must be True to proceed)

        Returns:
            Dictionary with deletion results
        """
        await ctx.info(f"Request to delete source: {source_url}")

        if not confirm:
            raise ToolError(
                "Deletion requires confirmation. Set confirm=True to proceed."
            )

        try:
            # Delete from RAG service (which handles Qdrant deletion)
            async with RagService() as rag_service:
                rag_deleted = await rag_service.delete_source(source_url)

            # Source registry is no longer maintained separately
            # All source tracking is now handled through Qdrant

            result = {
                "source_url": source_url,
                "documents_deleted": rag_deleted,
                "success": rag_deleted > 0,
                "timestamp": datetime.utcnow().isoformat(),
            }

            if result["success"]:
                await ctx.info(
                    f"Successfully deleted {rag_deleted} documents for source: {source_url}"
                )
            else:
                await ctx.info(f"No documents found to delete for source: {source_url}")

            return result

        except Exception as e:
            error_msg = f"Failed to delete source: {e!s}"
            await ctx.info(error_msg)
            raise ToolError(error_msg) from e
