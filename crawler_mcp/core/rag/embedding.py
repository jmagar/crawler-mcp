"""
Embedding generation pipeline with parallel processing for RAG operations.

This module provides high-performance embedding generation using parallel workers,
queue management, and optimized batch processing for maximum throughput.
"""

import asyncio
import logging
import time
from asyncio import Queue, Semaphore
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...config import settings
from ...core.embeddings import EmbeddingService
from ...core.vectors import VectorService
from ...models.rag import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingProcessResult:
    """Result of an embedding operation with success/failure information."""

    success: bool
    embedding: list[float] | None = None
    error: str | None = None


class EmbeddingCache:
    """Caching layer for embedding operations."""

    def __init__(self, max_size: int = 10000):
        self.cache: dict[str, list[float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_size = max_size

    def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        """
        Get cached embedding for a text hash.

        Args:
            text_hash: Hash of the text content

        Returns:
            Cached embedding vector or None
        """
        if text_hash in self.cache:
            self.cache_hits += 1
            return self.cache[text_hash]

        self.cache_misses += 1
        return None

    def cache_embedding(self, text_hash: str, embedding: list[float]) -> None:
        """
        Cache an embedding for future use.

        Args:
            text_hash: Hash of the text content
            embedding: Embedding vector to cache
        """
        # If cache is full, remove oldest entry (simple FIFO)
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[text_hash] = embedding

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_statistics(self) -> dict[str, int | float]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.max_size,
        }


class EmbeddingWorker:
    """Individual embedding worker for parallel processing."""

    def __init__(self, worker_id: int, cache: EmbeddingCache | None = None):
        self.worker_id = worker_id
        self.cache = cache
        self.embedding_service: EmbeddingService | None = None
        self.processed_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the embedding service for this worker."""
        if not self._initialized:
            self.embedding_service = EmbeddingService()
            if self.embedding_service is not None:
                await self.embedding_service.__aenter__()
            self._initialized = True
            logger.debug(f"Embedding worker {self.worker_id} initialized")

    async def process_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Process a batch of texts to generate embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Check cache for existing embeddings
            import hashlib

            cached_embeddings: list[list[float] | None] = []
            uncached_texts: list[str | None] = []
            text_hashes = []

            for text in texts:
                # Generate stable hash of normalized text content
                # Safely handle text that might be a list
                if isinstance(text, list):
                    text_str = "\n".join(str(item) for item in text)
                else:
                    text_str = str(text)
                normalized_text = text_str.strip().lower()
                text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
                text_hashes.append(text_hash)

                cached_embedding = None
                if self.cache is not None:
                    cached_embedding = self.cache.get_cached_embedding(text_hash)

                if cached_embedding is not None:
                    cached_embeddings.append(cached_embedding)
                    uncached_texts.append(None)  # Placeholder for cached item
                else:
                    cached_embeddings.append(None)  # Placeholder for uncached item
                    uncached_texts.append(text)

            # Generate embeddings for uncached texts only
            if any(text is not None for text in uncached_texts):
                texts_to_embed = [text for text in uncached_texts if text is not None]
                embedding_results = []

                # Use true batch processing for maximum speed if service is available
                if self.embedding_service is not None:
                    if len(texts_to_embed) <= 512:  # Our extreme true batch threshold
                        embedding_results = (
                            await self.embedding_service.generate_embeddings_true_batch(
                                texts_to_embed
                            )
                        )
                    else:
                        embedding_results = (
                            await self.embedding_service.generate_embeddings_batch(
                                texts_to_embed, batch_size=settings.tei_batch_size
                            )
                        )

                # Extract embedding vectors and cache them
                new_embeddings = [result.embedding for result in embedding_results]
                embedding_idx = 0

                for i, (text_candidate, text_hash) in enumerate(
                    zip(uncached_texts, text_hashes, strict=False)
                ):
                    current_text: str | None = text_candidate
                    if current_text is not None:  # This was an uncached text
                        embedding = new_embeddings[embedding_idx]
                        cached_embeddings[i] = embedding
                        if self.cache is not None:
                            self.cache.cache_embedding(text_hash, embedding)
                        embedding_idx += 1

            self.processed_count += len(texts)
            self.total_processing_time += time.time() - start_time

            cache_hits = self.cache.cache_hits if self.cache is not None else 0
            cache_misses = self.cache.cache_misses if self.cache is not None else 0
            logger.debug(
                f"Worker {self.worker_id} processed batch of {len(texts)} texts "
                f"(cache hits: {cache_hits}, misses: {cache_misses})"
            )

            # Ensure all None values are replaced with valid embeddings
            final_embeddings = []
            for emb in cached_embeddings:
                if emb is not None:
                    final_embeddings.append(emb)
                else:
                    # This shouldn't happen if logic is correct, but provide fallback
                    final_embeddings.append([0.0] * 384)  # Common embedding dimension
            return final_embeddings

        except Exception as e:
            self.error_count += 1
            logger.error(f"Worker {self.worker_id} failed to process batch: {e}")
            # Return empty embeddings for failed texts
            return [[] for _ in texts]

    async def shutdown(self) -> None:
        """Shutdown the worker and clean up resources."""
        if self._initialized and self.embedding_service:
            await self.embedding_service.__aexit__(None, None, None)
            self._initialized = False
            logger.debug(f"Embedding worker {self.worker_id} shutdown")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get worker performance statistics.

        Returns:
            Dictionary with worker statistics
        """
        avg_processing_time = (
            self.total_processing_time / self.processed_count
            if self.processed_count > 0
            else 0
        )

        return {
            "worker_id": self.worker_id,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": avg_processing_time,
            "success_rate": (
                (self.processed_count - self.error_count) / self.processed_count
                if self.processed_count > 0
                else 1.0
            ),
        }


class EmbeddingPipeline:
    """High-performance embedding generation pipeline."""

    def __init__(self) -> None:
        self.embedding_service: EmbeddingService | None = None
        self.vector_service: VectorService | None = None
        self.workers: list[Any] = []
        self.cache = EmbeddingCache(max_size=1000)  # Default cache size
        self.batch_size = settings.tei_batch_size
        self.max_workers = settings.embedding_workers
        self._pipeline_running = False
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the embedding pipeline."""
        if not self._initialized:
            self.embedding_service = EmbeddingService()
            self.vector_service = VectorService()
            if self.embedding_service is not None:
                await self.embedding_service.__aenter__()
            if self.vector_service is not None:
                await self.vector_service.__aenter__()
            self._initialized = True
            logger.info(
                f"Embedding pipeline initialized with {self.max_workers} workers"
            )

    async def shutdown(self) -> None:
        """Shutdown the embedding pipeline."""
        if self._pipeline_running:
            await self.stop_pipeline()

        if self._initialized:
            # Shutdown all workers
            for worker in self.workers:
                await worker.shutdown()
            self.workers.clear()

            # Shutdown services
            if self.embedding_service:
                await self.embedding_service.__aexit__(None, None, None)  # type: ignore[arg-type]
            if self.vector_service:
                await self.vector_service.__aexit__(None, None, None)  # type: ignore[arg-type]

            self._initialized = False
            logger.info("Embedding pipeline shutdown complete")

    async def start_pipeline(self) -> None:
        """Start the embedding pipeline with worker pool."""
        if not self._initialized:
            await self.initialize()

        if not self._pipeline_running:
            # Create worker pool with shared cache
            self.workers = [
                EmbeddingWorker(worker_id, self.cache)
                for worker_id in range(self.max_workers)
            ]

            # Initialize all workers
            for worker in self.workers:
                await worker.initialize()

            self._pipeline_running = True
            logger.info(f"Embedding pipeline started with {len(self.workers)} workers")

    async def stop_pipeline(self) -> None:
        """Stop the embedding pipeline."""
        if self._pipeline_running:
            # Shutdown all workers
            for worker in self.workers:
                await worker.shutdown()

            self._pipeline_running = False
            logger.info("Embedding pipeline stopped")

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Optional batch size override

        Returns:
            List of embedding vectors
        """
        if not self._pipeline_running:
            await self.start_pipeline()

        if not texts:
            return []

        effective_batch_size = batch_size or self.batch_size

        # Use single batch processing if within limits
        if len(texts) <= effective_batch_size:
            if self.workers:
                return await self.workers[0].process_batch(texts)  # type: ignore[no-any-return]
            else:
                # Fallback to direct service call
                if self.embedding_service is not None:
                    embedding_results = (
                        await self.embedding_service.generate_embeddings_batch(
                            texts, batch_size=effective_batch_size
                        )
                    )
                    return [result.embedding for result in embedding_results]
                else:
                    # Return empty embeddings if service unavailable
                    return [[0.0] * 768] * len(texts)

        # Use parallel batch processing for larger sets
        process_results: list[
            EmbeddingProcessResult
        ] = await self._process_batches_parallel(texts, effective_batch_size)
        return [
            result.embedding if result.success and result.embedding else [0.0] * 768
            for result in process_results
        ]

    async def process_chunks_parallel(
        self,
        chunks: list[DocumentChunk],
        progress_callback: Callable[[int, int], None] | None = None,
        base_progress: int = 0,
    ) -> list[DocumentChunk]:
        """
        Process document chunks in parallel to generate embeddings.

        Args:
            chunks: List of document chunks to process
            progress_callback: Optional progress callback
            base_progress: Base progress offset for reporting

        Returns:
            List of chunks with embeddings attached
        """
        if not chunks:
            return []

        if not self._pipeline_running:
            await self.start_pipeline()

        start_time = time.time()

        # Use the parallel pipeline processing for maximum performance
        total_embeddings = await self._process_embeddings_pipeline(
            chunks, progress_callback, base_progress
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(
            f"Generated {total_embeddings} embeddings in {duration:.2f}s - "
            f"{total_embeddings / duration:.1f} embeddings/sec (parallel pipeline)"
        )

        return chunks

    async def _process_batches_parallel(
        self, texts: list[str], batch_size: int
    ) -> list[EmbeddingProcessResult]:
        """Process texts in parallel batches using worker pool."""
        # Split texts into batches
        text_batches = [
            texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
        ]

        # Process batches in parallel using available workers
        batch_tasks = []
        for i, batch in enumerate(text_batches):
            worker = self.workers[i % len(self.workers)]
            batch_tasks.append(worker.process_batch(batch))

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Flatten results
        all_embeddings: list[EmbeddingProcessResult] = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
                # Mark failed items clearly so upstream can decide to skip them
                batch_texts = text_batches[i]
                all_embeddings.extend(
                    [
                        EmbeddingProcessResult(success=False, error=str(result))
                        for _ in batch_texts
                    ]
                )
            else:
                # Convert successful embeddings to EmbeddingResult instances
                if not isinstance(result, BaseException):
                    all_embeddings.extend(
                        [
                            EmbeddingProcessResult(success=True, embedding=embedding)
                            for embedding in result
                        ]
                    )

        return all_embeddings

    async def _process_embeddings_pipeline(
        self,
        document_chunks: list[DocumentChunk],
        progress_callback: Callable[[int, int], None] | None = None,
        base_progress: int = 0,
        persist: bool = False,
    ) -> int:
        """
        Process embeddings using parallel pipeline for maximum performance.

        Uses multiple concurrent workers to:
        1. Generate embeddings in parallel batches
        2. Store vectors concurrently as they're ready
        3. Provide real-time progress feedback

        Args:
            document_chunks: List of document chunks to process
            progress_callback: Optional progress callback function
            base_progress: Base progress offset for reporting

        Returns:
            Total number of embeddings generated
        """
        if not document_chunks:
            return 0

        # Configure pipeline parameters based on our optimized settings
        max_concurrent_embedding_batches = min(
            self.max_workers, (len(document_chunks) // 64) + 1
        )  # Configurable parallel embedding workers
        max_concurrent_storage_ops = 8  # 8 parallel storage operations
        embedding_batch_size = self.batch_size  # 64 from our extreme config

        # Create queues for pipeline stages
        storage_queue: Queue[tuple[int, list[DocumentChunk]]] = Queue(
            maxsize=max_concurrent_storage_ops * 2
        )

        # Semaphores to control concurrency
        embedding_semaphore = Semaphore(max_concurrent_embedding_batches)
        storage_semaphore = Semaphore(max_concurrent_storage_ops)

        # Split chunks into batches for embedding
        chunk_batches = [
            document_chunks[i : i + embedding_batch_size]
            for i in range(0, len(document_chunks), embedding_batch_size)
        ]

        # Create exactly one storage worker per chunk batch to ensure all items are processed
        num_storage_workers = len(chunk_batches)

        logger.info(
            f"Starting parallel pipeline: {len(chunk_batches)} embedding batches, "
            f"{max_concurrent_embedding_batches} embedding workers, "
            f"{num_storage_workers} storage workers"
        )

        # Progress tracking
        embeddings_completed = 0
        storage_completed = 0
        total_chunks = len(document_chunks)

        async def embedding_worker(
            batch_id: int, chunk_batch: list[DocumentChunk]
        ) -> None:
            """Worker to generate embeddings for a batch of chunks."""
            async with embedding_semaphore:
                try:
                    texts = [chunk.content for chunk in chunk_batch]

                    # Use worker pool for processing
                    worker = self.workers[batch_id % len(self.workers)]
                    embeddings = await worker.process_batch(texts)

                    # Attach embeddings to chunks
                    for chunk, embedding in zip(chunk_batch, embeddings, strict=False):
                        chunk.embedding = embedding

                    # Queue batch for storage
                    await storage_queue.put((batch_id, chunk_batch))

                    nonlocal embeddings_completed
                    embeddings_completed += len(chunk_batch)

                    logger.debug(
                        f"Embedding batch {batch_id} completed: {len(chunk_batch)} chunks"
                    )

                except Exception as e:
                    logger.error(f"Error in embedding worker {batch_id}: {e}")
                    # Still queue for storage with empty embeddings to maintain progress
                    await storage_queue.put((batch_id, chunk_batch))

        async def storage_worker() -> None:
            """Worker to store embedded chunks as they become available."""
            while True:
                batch_id, chunk_batch = await storage_queue.get()
                try:
                    # Use semaphore only for the actual storage operation
                    async with storage_semaphore:
                        # Filter chunks with valid embeddings
                        valid_chunks = [
                            chunk
                            for chunk in chunk_batch
                            if chunk.embedding is not None and len(chunk.embedding) > 0
                        ]

                        if valid_chunks and persist and self.vector_service is not None:
                            # Store chunks in Qdrant concurrently (only if persist=True)
                            await self.vector_service.upsert_documents(valid_chunks)

                    nonlocal storage_completed
                    storage_completed += len(chunk_batch)

                    # Update progress
                    if progress_callback:
                        progress = base_progress + int(
                            (storage_completed / total_chunks) * 10
                        )  # 10 units for embeddings
                        progress_callback(
                            progress,
                            base_progress + 10,
                        )

                    logger.debug(
                        f"Storage batch {batch_id} completed: {len(valid_chunks)} chunks stored"
                    )

                except Exception as e:
                    logger.error(f"Error in storage worker: {e}")
                finally:
                    # Always call task_done() to prevent queue.join() from hanging
                    storage_queue.task_done()

        # Create all embedding worker tasks at once for true parallelism
        embedding_tasks = [
            embedding_worker(batch_id, chunk_batch)
            for batch_id, chunk_batch in enumerate(chunk_batches)
        ]

        # Start storage workers in background
        storage_tasks = [
            asyncio.create_task(storage_worker()) for _ in range(num_storage_workers)
        ]

        # Wait for all embedding tasks to complete
        await asyncio.gather(*embedding_tasks, return_exceptions=True)

        # Wait for all queued items to be processed
        await storage_queue.join()

        # Cancel storage workers (they're now idle)
        for task in storage_tasks:
            task.cancel()

        # Wait for cancellation to complete
        await asyncio.gather(*storage_tasks, return_exceptions=True)

        logger.info(
            f"Parallel pipeline completed: {embeddings_completed} embeddings generated, "
            f"{storage_completed} chunks processed"
        )

        return embeddings_completed

    def optimize_batch_size(self, total_items: int) -> int:
        """
        Optimize batch size based on total items and available workers.

        Args:
            total_items: Total number of items to process

        Returns:
            Optimized batch size
        """
        if total_items <= self.batch_size:
            return total_items

        # Calculate optimal batch size to distribute work evenly
        optimal_batches = min(total_items // self.max_workers, self.batch_size)
        return max(optimal_batches, 1)

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of the embedding pipeline.

        Returns:
            Health status dictionary
        """
        health_status = {
            "pipeline_running": self._pipeline_running,
            "workers_count": len(self.workers),
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "cache_stats": self.cache.get_cache_statistics(),
        }

        # Check worker health
        if self.workers:
            worker_stats = [worker.get_statistics() for worker in self.workers]
            health_status["worker_stats"] = worker_stats

            # Calculate aggregate statistics
            total_processed = sum(stats["processed_count"] for stats in worker_stats)
            total_errors = sum(stats["error_count"] for stats in worker_stats)

            health_status["aggregate_stats"] = {
                "total_processed": total_processed,
                "total_errors": total_errors,
                "overall_success_rate": (
                    (total_processed - total_errors) / total_processed
                    if total_processed > 0
                    else 1.0
                ),
            }

        # Check embedding service health
        if self.embedding_service:
            embedding_health = await self.embedding_service.health_check()
            health_status["embedding_service"] = (
                "healthy" if embedding_health else "unhealthy"
            )

        return health_status
