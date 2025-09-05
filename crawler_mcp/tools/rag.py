"""
FastMCP tools for RAG operations (optimized namespace).
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from crawler_mcp.core import RagService, VectorService
from crawler_mcp.middleware.progress import progress_middleware
from crawler_mcp.models.rag import RagQuery

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
        await ctx.info(f"Performing RAG query: '{query}' (limit: {limit})")

        if limit > 100:
            raise ToolError("limit cannot exceed 100")
        if not 0.0 <= min_score <= 1.0:
            raise ToolError("min_score must be between 0.0 and 1.0")
        if not query.strip():
            raise ToolError("query cannot be empty")

        progress_tracker = progress_middleware.create_tracker(f"rag_{hash(query)}")
        try:
            await ctx.report_progress(progress=1, total=4)
            rag_query_obj = RagQuery(
                query=query.strip(),
                limit=limit,
                min_score=min_score,
                source_filters=source_filters,
                include_content=include_content,
                rerank=rerank,
            )

            await ctx.report_progress(progress=2, total=4)
            async with RagService() as rag_service:
                await ctx.report_progress(progress=3, total=4)
                rag_result = await rag_service.query(rag_query_obj, rerank=rerank)

            await ctx.report_progress(progress=4, total=4)

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
                if include_content:
                    match_data["document"]["content"] = match.document.content
                    if match.highlighted_content:
                        match_data["highlighted_content"] = match.highlighted_content
                if match.document.metadata:
                    match_data["document"]["metadata"] = match.document.metadata
                result["matches"].append(match_data)

            await ctx.info(
                f"RAG query completed: {rag_result.total_matches} matches in {rag_result.processing_time:.3f}s "
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
        await ctx.info(f"Listing sources (limit: {limit}, offset: {offset})")

        if limit > 200:
            raise ToolError("limit cannot exceed 200")
        if offset < 0:
            raise ToolError("offset cannot be negative")

        try:
            async with VectorService() as vector_service:
                sources_response = await vector_service.get_unique_sources(
                    domains=domains,
                    search_term=search_term,
                    limit=limit,
                    offset=offset,
                )
                vector_stats = await vector_service.get_sources_stats(lightweight=True)

            result: dict[str, Any] = {
                "sources": [],
                "pagination": sources_response["pagination"],
                "statistics": {
                    "vector_database": vector_stats,
                    "source_registry": {
                        "registered_sources": 0,
                        "sources_by_type": {},
                        "sources_by_status": {},
                        "stale_sources": 0,
                    },
                },
                "filters_applied": {
                    "source_types": source_types,
                    "domains": domains,
                    "statuses": statuses,
                    "search_term": search_term,
                },
            }

            from urllib.parse import urlparse

            for source in sources_response["sources"]:
                parsed_url = urlparse(source["url"])
                source_data = {
                    "id": f"src_{hash(source['url']) & 0x7FFFFFFF:08x}",
                    "url": source["url"],
                    "title": source["title"],
                    "source_type": source["source_type"],
                    "domain": parsed_url.netloc,
                    "path": parsed_url.path,
                }
                result["sources"].append(source_data)

            return result
        except Exception as e:
            raise ToolError(f"Failed to list sources: {e!s}") from e
