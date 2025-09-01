"""
Service orchestration and query processing for RAG operations.

This module provides the main RagService class that coordinates all RAG operations
including query processing, caching, reranking, and service lifecycle management.
"""

import asyncio
import hashlib
import logging
import math
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from threading import RLock
from typing import Any

from fastmcp.exceptions import ToolError

from ...config import settings
from ...models.crawl import CrawlResult
from ...models.rag import DocumentChunk, RagQuery, RagResult, SearchMatch
from ..embeddings import EmbeddingService
from ..utils import normalize_url
from ..vectors import VectorService
from .processing import ProcessingPipeline

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Simple in-memory cache for RAG query results with TTL support.
    """

    def __init__(self, max_size: int = 1000, ttl_minutes: int = 15):
        self.cache: dict[str, tuple[RagResult, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self._lock = RLock()

    def _generate_cache_key(
        self,
        query: str,
        limit: int,
        min_score: float,
        source_filters: list[str] | None,
        rerank: bool,
        include_content: bool,
        include_metadata: bool,
        date_range: tuple[str, str] | None,
    ) -> str:
        """Generate a deterministic cache key from query parameters."""
        key_components = [
            str(query).strip().lower(),
            str(limit),
            str(min_score),
            str(sorted(source_filters) if source_filters else ""),
            str(rerank),
            str(include_content),
            str(include_metadata),
            str(date_range) if date_range else "",
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self,
        query: str,
        limit: int,
        min_score: float,
        source_filters: list[str] | None,
        rerank: bool,
        include_content: bool,
        include_metadata: bool,
        date_range: tuple[str, str] | None,
    ) -> RagResult | None:
        """Get cached result if it exists and hasn't expired."""
        cache_key = self._generate_cache_key(
            query,
            limit,
            min_score,
            source_filters,
            rerank,
            include_content,
            include_metadata,
            date_range,
        )

        with self._lock:
            if cache_key in self.cache:
                result, timestamp = self.cache[cache_key]
                if datetime.utcnow() - timestamp < self.ttl:
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return result
                else:
                    # Expired, remove from cache
                    del self.cache[cache_key]
                    logger.debug(f"Cache expired for query: {query[:50]}...")

        return None

    def put(
        self,
        query: str,
        limit: int,
        min_score: float,
        source_filters: list[str] | None,
        rerank: bool,
        result: RagResult,
        include_content: bool,
        include_metadata: bool,
        date_range: tuple[str, str] | None,
    ) -> None:
        """Cache a result with current timestamp."""
        cache_key = self._generate_cache_key(
            query,
            limit,
            min_score,
            source_filters,
            rerank,
            include_content,
            include_metadata,
            date_range,
        )

        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug("Cache full, removed oldest entry")

        self.cache[cache_key] = (result, datetime.utcnow())
        logger.debug(f"Cached result for query: {query[:50]}...")

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        logger.info("Query cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = datetime.utcnow()
        valid_entries = sum(
            1 for _, timestamp in self.cache.values() if now - timestamp < self.ttl
        )

        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries,
            "max_size": self.max_size,
            "ttl_minutes": self.ttl.total_seconds() / 60,
        }


class ServiceMetrics:
    """Tracks service-level metrics and performance."""

    def __init__(self) -> None:
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_query_time = 0.0
        self.rerank_count = 0
        self.embedding_errors = 0
        self.vector_errors = 0

    def record_query(
        self, cache_hit: bool, query_time: float, reranked: bool = False
    ) -> None:
        """Record query metrics."""
        self.query_count += 1
        self.total_query_time += query_time

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if reranked:
            self.rerank_count += 1

    def record_embedding_error(self) -> None:
        """Record embedding service error."""
        self.embedding_errors += 1

    def record_vector_error(self) -> None:
        """Record vector service error."""
        self.vector_errors += 1

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive service metrics."""
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0.0
        )

        avg_query_time = (
            self.total_query_time / self.query_count if self.query_count > 0 else 0.0
        )

        return {
            "total_queries": self.query_count,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "average_query_time": avg_query_time,
            "total_query_time": self.total_query_time,
            "rerank_usage": self.rerank_count,
            "embedding_errors": self.embedding_errors,
            "vector_errors": self.vector_errors,
        }


class RagService:
    """
    Service for RAG operations combining embedding generation and vector search.
    Uses singleton pattern to keep models loaded in memory for optimal performance.
    """

    _instance = None
    _initialized = False
    _context_count = 0
    _lock = None
    _auto_opened = False

    def __new__(cls) -> "RagService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Prevent re-initialization of models
        if self._initialized:
            return

        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()
        self.processing_pipeline = ProcessingPipeline()

        # Initialize query cache for performance optimization
        self.query_cache = QueryCache(max_size=1000, ttl_minutes=15)

        # Initialize service metrics
        self.metrics = ServiceMetrics()

        # Add missing attributes from settings
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.tokenizer = None
        self.tokenizer_type = "tiktoken"  # Default tokenizer type

        # Initialize tokenizer
        try:
            import tiktoken

            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.tokenizer_type = "tiktoken"
        except ImportError:
            logger.warning("tiktoken not available, using character-based chunking")
            self.tokenizer_type = "character"

        # Initialize Qwen3 reranker with GPU optimization
        self.reranker = None
        self.reranker_type = "none"

        if settings.reranker_enabled:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                logger.warning(
                    "sentence-transformers not installed. Reranking disabled: %s", e
                )
            else:
                try:
                    # Force GPU usage for reranker if available
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    self.reranker = CrossEncoder(settings.reranker_model, device=device)
                    self.reranker_type = "qwen3"
                    logger.info(
                        "Using Qwen3 reranker: %s on device: %s",
                        settings.reranker_model,
                        device,
                    )
                except (OSError, ValueError) as e:
                    logger.warning(
                        "Failed to load Qwen3 reranker %s: %s",
                        settings.reranker_model,
                        e,
                    )
                    if settings.reranker_fallback_to_custom:
                        self.reranker_type = "custom"
                        logger.info("Using custom reranking algorithm as fallback")
                    else:
                        self.reranker_type = "none"
                        logger.info("Reranking disabled due to model loading failure")

        # Initialize lock for context manager reference counting
        if RagService._lock is None:
            RagService._lock = asyncio.Lock()

        # Mark as initialized to prevent reloading models
        RagService._initialized = True

    async def __aenter__(self) -> "RagService":
        """Async context manager entry with reference counting."""
        if RagService._lock is None:
            RagService._lock = asyncio.Lock()
        async with RagService._lock:
            self._context_count += 1
            # Only initialize underlying services on first context entry
            if self._context_count == 1:
                logger.debug("Initializing underlying services (first context)")
                await self.embedding_service.__aenter__()
                await self.vector_service.__aenter__()
            else:
                logger.debug(f"Reusing services (context count: {self._context_count})")
        return self

    async def __aexit__(
        self,
        exc_type: type,
        exc_val: Exception,
        exc_tb: object,
    ) -> None:
        """Async context manager exit with reference counting."""
        if RagService._lock is None:
            RagService._lock = asyncio.Lock()
        async with RagService._lock:
            self._context_count -= 1
            # Only close underlying services when last context exits
            if self._context_count == 0:
                logger.debug("Closing underlying services (last context)")
                await self.embedding_service.__aexit__(exc_type, exc_val, exc_tb)
                await self.vector_service.__aexit__(exc_type, exc_val, exc_tb)
                RagService._auto_opened = False
            else:
                logger.debug(
                    f"Keeping services alive (context count: {self._context_count})"
                )

    async def close(self) -> None:
        """Close all services."""
        await self.embedding_service.close()
        await self.vector_service.close()

    @classmethod
    async def reset(cls) -> None:
        """Reset the singleton for testing purposes."""
        if cls._instance is not None:
            # Close any open services
            await cls._instance.close()
            # Reset singleton state
            cls._instance = None
            cls._initialized = False
            cls._context_count = 0
            cls._lock = None
            cls._auto_opened = False

    async def _ensure_open(self) -> None:
        """Ensure underlying services are initialized without requiring context manager."""
        if RagService._lock is None:
            RagService._lock = asyncio.Lock()
        async with RagService._lock:
            # Only initialize if not already done
            if self._context_count == 0:
                logger.debug("Auto-initializing underlying services (ensure_open)")
                await self.embedding_service.__aenter__()
                await self.vector_service.__aenter__()
                RagService._auto_opened = True
            self._context_count += 1

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all dependent services.

        Returns:
            Dictionary with service health status
        """
        return {
            "embedding_service": await self.embedding_service.health_check(),
            "vector_service": await self.vector_service.health_check(),
        }

    # Legacy helper methods for backwards compatibility
    def _find_paragraph_boundary(self, search_text: str, ideal_end: int) -> int | None:
        """Find paragraph break boundary."""
        from .chunking import find_paragraph_boundary

        return find_paragraph_boundary(search_text, ideal_end)

    def _find_sentence_boundary(self, search_text: str, ideal_end: int) -> int | None:
        """Find sentence ending boundary."""
        from .chunking import find_sentence_boundary

        return find_sentence_boundary(search_text, ideal_end)

    # Deduplication helper methods for backwards compatibility
    def _generate_deterministic_id(self, url: str, chunk_index: int) -> str:
        """
        Generate deterministic ID from URL and chunk index.

        Args:
            url: Source URL
            chunk_index: Index of the chunk within the document

        Returns:
            Deterministic UUID string
        """
        import uuid

        normalized_url = normalize_url(url)
        id_string = f"{normalized_url}:{chunk_index}"
        # Generate a deterministic UUID from the hash
        hash_bytes = hashlib.sha256(id_string.encode()).digest()[:16]
        # Create UUID from the first 16 bytes of the hash
        deterministic_uuid = uuid.UUID(bytes=hash_bytes)
        return str(deterministic_uuid)

    def _calculate_content_hash(self, content: str) -> str:
        """
        Calculate SHA256 hash of content for change detection.

        Args:
            content: Text content to hash

        Returns:
            SHA256 hash hexdigest string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Backwards compatibility helper methods
    def _is_random_uuid(self, chunk_id: str) -> bool:
        """
        Check if a chunk ID appears to be a random UUID.

        This helps identify chunks that were created before deterministic IDs
        were implemented.

        Args:
            chunk_id: Chunk ID to check

        Returns:
            True if the ID appears to be a random UUID
        """
        # Pattern for random UUIDs (32+ hex characters, possibly with dashes)
        uuid_pattern = re.compile(r"^[0-9a-f\-]{32,}$", re.IGNORECASE)
        clean_id = chunk_id.replace("-", "")
        return bool(uuid_pattern.match(clean_id))

    async def process_crawl_result(
        self,
        crawl_result: CrawlResult,
        deduplication: bool | None = None,
        force_update: bool = False,
        progress_callback: Callable[..., None] | None = None,
        seed_url: str | None = None,
        crawl_session_id: str | None = None,
    ) -> dict[str, int]:
        """
        Delegate to processing pipeline for crawl result processing.

        Args:
            crawl_result: Crawl result to process
            deduplication: Enable deduplication (uses settings default if None)
            force_update: Force update existing chunks
            progress_callback: Optional progress callback
            seed_url: Original URL that initiated the crawl
            crawl_session_id: Unique ID for this crawl session

        Returns:
            Processing statistics
        """
        return await self.processing_pipeline.process_crawl_result(
            crawl_result,
            deduplication,
            force_update,
            progress_callback,
            seed_url,
            crawl_session_id,
        )

    async def query(self, query: RagQuery, rerank: bool = True) -> RagResult:
        """
        Perform a RAG query to find relevant documents.

        Args:
            query: The RAG query parameters
            rerank: Whether to apply re-ranking to results

        Returns:
            RagResult with matched documents
        """
        await self._ensure_open()
        start_time = time.time()

        # Compute effective rerank flag
        effective_rerank = rerank and getattr(query, "rerank", True)

        # Check cache first for exact query match
        # Convert datetime tuple to string tuple for cache key if needed
        date_range_str = None
        if query.date_range:
            date_range_str = (
                query.date_range[0].isoformat(),
                query.date_range[1].isoformat(),
            )

        cached_result = self.query_cache.get(
            query.query,
            query.limit,
            query.min_score,
            query.source_filters,
            rerank,
            include_content=query.include_content,
            include_metadata=query.include_metadata,
            date_range=date_range_str,
        )
        if cached_result is not None:
            logger.info(f"Cache hit for query: '{query.query[:50]}...'")
            self.metrics.record_query(
                cache_hit=True,
                query_time=time.time() - start_time,
                reranked=effective_rerank,
            )
            return cached_result

        try:
            # Generate query embedding
            embedding_start = time.time()
            try:
                embedding_result = await self.embedding_service.generate_embedding(
                    query.query
                )
            except Exception:
                self.metrics.record_embedding_error()
                raise
            embedding_time = time.time() - embedding_start

            # Search vector database
            search_start = time.time()
            try:
                search_matches = await self.vector_service.search_similar(
                    query_vector=embedding_result.embedding,
                    limit=query.limit,
                    score_threshold=query.min_score,
                    source_filter=query.source_filters,
                    date_range=query.date_range,
                )
            except Exception:
                self.metrics.record_vector_error()
                raise
            search_time = time.time() - search_start

            # Apply re-ranking if requested
            rerank_time = None
            if (
                effective_rerank
                and len(search_matches) > 1
                and self.reranker_type != "none"
            ):
                rerank_start = time.time()
                search_matches = await self._rerank_results(query.query, search_matches)
                rerank_time = time.time() - rerank_start

            # Filter matches based on query parameters
            filtered_matches = []
            for match in search_matches:
                # Apply content/metadata filters if specified
                if not query.include_content:
                    match.document.content = ""  # Remove content to save bandwidth

                if not query.include_metadata:
                    match.document.metadata = {}

                filtered_matches.append(match)

            processing_time = time.time() - start_time

            result = RagResult(
                query=query.query,
                matches=filtered_matches,
                total_matches=len(filtered_matches),
                processing_time=processing_time,
                embedding_time=embedding_time,
                search_time=search_time,
                rerank_time=rerank_time,
            )

            logger.info(
                f"RAG query completed: {len(filtered_matches)} matches in {processing_time:.3f}s "
                f"(embed: {embedding_time:.3f}s, search: {search_time:.3f}s"
                f"{f', rerank: {rerank_time:.3f}s' if rerank_time else ''})"
            )

            # Record metrics
            self.metrics.record_query(
                cache_hit=False,
                query_time=processing_time,
                reranked=effective_rerank and rerank_time is not None,
            )

            # Cache the result for future queries
            self.query_cache.put(
                query.query,
                query.limit,
                query.min_score,
                query.source_filters,
                rerank,
                result,
                include_content=query.include_content,
                include_metadata=query.include_metadata,
                date_range=date_range_str,
            )

            return result

        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            raise ToolError(f"RAG query failed: {e!s}") from e

    async def _rerank_results(
        self,
        query: str,
        matches: list[SearchMatch],
        top_k: int | None = None,
    ) -> list[SearchMatch]:
        """
        Re-rank search results using Qwen3 reranker or custom algorithm.

        Args:
            query: Original query text
            matches: Initial search matches
            top_k: Number of top results to return (optional)

        Returns:
            Re-ranked list of search matches
        """
        if not matches:
            return matches

        try:
            if self.reranker_type == "qwen3":
                return await self._rerank_with_qwen3(query, matches, top_k)

            if self.reranker_type == "custom":
                return await self._rerank_with_custom_algorithm(query, matches, top_k)

            return matches  # No reranking

        except Exception as e:
            logger.warning("Re-ranking failed, returning original results: %s", e)
            return matches

    async def _rerank_with_qwen3(
        self,
        query: str,
        matches: list[SearchMatch],
        top_k: int | None = None,
    ) -> list[SearchMatch]:
        """
        Re-rank using Qwen3 CrossEncoder reranker.
        """
        if not self.reranker or not matches:
            return matches

        try:
            # Prepare query-document pairs for the reranker
            pairs = []
            for match in matches:
                # Truncate content to reranker max length
                content = match.document.content[: settings.reranker_max_length]
                pairs.append([query, content])

            # Get reranking scores from Qwen3 model using proper batching
            # Process all pairs in a single batch for efficiency
            scores = await asyncio.to_thread(self.reranker.predict, pairs)

            # Update match scores with reranker predictions
            for match, score in zip(matches, scores, strict=False):
                # Convert reranker logits to normalized probability scores
                reranker_score = float(score)

                # Apply sigmoid normalization for better score distribution
                normalized_reranker_score = 1.0 / (1.0 + math.exp(-reranker_score))

                # Combine scores with reranker taking priority (it's query-specific)
                # Keep original vector similarity but boost with reranker confidence
                original_score = match.score
                match.score = 0.4 * match.score + 0.6 * normalized_reranker_score

                # Recalculate relevance based on new score
                if match.score >= 0.8:
                    match.relevance = "high"
                elif match.score >= 0.6:
                    match.relevance = "medium"
                else:
                    match.relevance = "low"

                logger.debug(
                    "Reranking: raw_logit=%.4f, sigmoid_score=%.4f, original_score=%.4f, final_score=%.4f, reranker_model=%s",
                    reranker_score,
                    normalized_reranker_score,
                    original_score,
                    match.score,
                    settings.reranker_model,
                )

            # Sort by updated scores
            matches.sort(key=lambda m: m.score, reverse=True)

            # Return top_k if specified
            if top_k and top_k < len(matches):
                matches = matches[:top_k]

            logger.debug(f"Qwen3 reranked {len(matches)} results")
            return matches

        except Exception:
            logger.exception("Qwen3 reranking failed")
            # Fallback to custom algorithm if available
            if settings.reranker_fallback_to_custom:
                logger.info("Falling back to custom reranking algorithm")
                return await self._rerank_with_custom_algorithm(query, matches, top_k)
            return matches

    async def _rerank_with_custom_algorithm(
        self,
        query: str,
        matches: list[SearchMatch],
        top_k: int | None = None,
    ) -> list[SearchMatch]:
        """
        Re-rank using custom hybrid scoring algorithm (original implementation).
        """
        try:
            # Simple re-ranking based on keyword overlap and length
            query_words = set(query.lower().split())

            for match in matches:
                content_words = set(match.document.content.lower().split())

                # Keyword overlap score
                keyword_overlap = len(query_words.intersection(content_words)) / len(
                    query_words
                )

                # Length penalty (prefer moderate length chunks)
                optimal_length = 500  # characters
                length_penalty = (
                    1.0
                    - abs(len(match.document.content) - optimal_length) / optimal_length
                )
                length_penalty = max(0.1, min(1.0, length_penalty))

                # Title relevance bonus
                title_bonus = 0.0
                if match.document.source_title:
                    title_words = set(match.document.source_title.lower().split())
                    title_overlap = len(query_words.intersection(title_words)) / len(
                        query_words
                    )
                    title_bonus = title_overlap * 0.1

                # Combine scores (weighted average)
                combined_score = (
                    match.score * 0.7  # Vector similarity (primary)
                    + keyword_overlap * 0.2  # Keyword overlap
                    + length_penalty * 0.05  # Length preference
                    + title_bonus * 0.05  # Title relevance
                )

                match.score = min(1.0, combined_score)

                # Recalculate relevance to match the updated score (consistent with Qwen path)
                if match.score >= 0.8:
                    match.relevance = "high"
                elif match.score >= 0.6:
                    match.relevance = "medium"
                else:
                    match.relevance = "low"

            # Sort by combined score
            matches.sort(key=lambda m: m.score, reverse=True)

            # Return top_k if specified
            if top_k and top_k < len(matches):
                matches = matches[:top_k]

            logger.debug("Custom algorithm reranked %d results", len(matches))
            return matches

        except Exception as e:
            logger.warning("Custom reranking failed: %s", e)
            return matches

    async def get_stats(self) -> dict[str, Any]:
        """
        Get RAG system statistics.

        Returns:
            Dictionary with system statistics
        """
        await self._ensure_open()
        try:
            collection_info = await self.vector_service.get_collection_info()
            source_stats = await self.vector_service.get_sources_stats(lightweight=True)

            return {
                "collection": collection_info,
                "sources": source_stats,
                "cache": self.query_cache.get_stats(),
                "metrics": self.metrics.get_stats(),
                "config": {
                    "chunk_size_tokens": self.chunk_size
                    if self.tokenizer_type == "tiktoken"
                    else None,
                    "chunk_overlap_tokens": self.chunk_overlap
                    if self.tokenizer_type == "tiktoken"
                    else None,
                    "chunk_size_chars": self.chunk_size
                    if self.tokenizer_type == "character"
                    else None,
                    "chunk_overlap_chars": self.chunk_overlap
                    if self.tokenizer_type == "character"
                    else None,
                    "tokenizer_type": getattr(self, "tokenizer_type", "character"),
                    "reranker_type": getattr(self, "reranker_type", "none"),
                    "reranker_model": settings.reranker_model
                    if self.reranker_type == "qwen3"
                    else None,
                    "reranker_enabled": settings.reranker_enabled,
                    "embedding_model": settings.tei_model,
                    "vector_dimension": settings.qdrant_vector_size,
                    "distance_metric": settings.qdrant_distance,
                },
            }

        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
            return {"error": str(e)}

    async def delete_source(self, source_url: str) -> bool:
        """
        Delete all documents from a specific source.

        Args:
            source_url: URL of the source to delete

        Returns:
            True if deletion was successful
        """
        await self._ensure_open()
        try:
            deleted_count = await self.vector_service.delete_documents_by_source(
                source_url
            )
            logger.info(f"Deleted documents from source: {source_url}")
            # Ensure deleted_count is an integer for comparison
            return int(deleted_count) > 0

        except Exception as e:
            logger.error(f"Error deleting source {source_url}: {e}")
            return False

    # Legacy method delegation for backwards compatibility
    async def _process_embeddings_pipeline(
        self,
        document_chunks: list[DocumentChunk],
        progress_callback: Callable[[int, int], None] | None = None,
        base_progress: int = 0,
    ) -> int:
        """
        Process embeddings using parallel pipeline for maximum performance.
        Delegates to the embedding pipeline in processing module.

        Args:
            document_chunks: List of document chunks to process
            progress_callback: Optional progress callback function
            base_progress: Base progress offset for reporting

        Returns:
            Total number of embeddings generated
        """
        return await self.processing_pipeline.embedding_pipeline._process_embeddings_pipeline(
            document_chunks, progress_callback, base_progress
        )
