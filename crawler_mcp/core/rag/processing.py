"""
Main processing pipeline coordination and workflow management for RAG operations.

This module orchestrates the complete RAG processing workflow, integrating
chunking, deduplication, embedding generation, and vector storage.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ...config import settings
from ...core.vectors import VectorService
from ...models.crawl import CrawlResult, PageContent
from ...models.rag import DocumentChunk
from .chunking import AdaptiveChunker
from .deduplication import VectorDeduplicationManager
from .embedding import EmbeddingPipeline

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Advanced progress tracking for long-running operations."""

    def __init__(self) -> None:
        self.current_stage = "initialization"
        self.progress_data: dict[str, Any] = {}
        self.start_time: float | None = None
        self.stage_start_times: dict[str, float] = {}

    def start_stage(self, stage_name: str, total_items: int) -> None:
        """
        Start a new processing stage.

        Args:
            stage_name: Name of the stage
            total_items: Total number of items to process in this stage
        """
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = time.monotonic()
        self.progress_data[stage_name] = {
            "total_items": total_items,
            "completed_items": 0,
            "start_time": self.stage_start_times[stage_name],
            "status": "running",
        }

        if self.start_time is None:
            self.start_time = time.monotonic()

        logger.info(f"Started stage '{stage_name}' with {total_items} items")

    def update_progress(self, completed: int, message: str = "") -> None:
        """
        Update progress for the current stage.

        Args:
            completed: Number of completed items
            message: Optional progress message
        """
        if self.current_stage in self.progress_data:
            self.progress_data[self.current_stage]["completed_items"] = completed
            self.progress_data[self.current_stage]["message"] = message

            total = self.progress_data[self.current_stage]["total_items"]
            progress_percent = (completed / total * 100) if total > 0 else 0

            logger.debug(
                f"Stage '{self.current_stage}': {completed}/{total} "
                f"({progress_percent:.1f}%) - {message}"
            )

    def complete_stage(self, stage_name: str) -> None:
        """
        Mark a stage as completed.

        Args:
            stage_name: Name of the stage to complete
        """
        if stage_name in self.progress_data:
            self.progress_data[stage_name]["status"] = "completed"
            self.progress_data[stage_name]["end_time"] = time.monotonic()

            duration = (
                self.progress_data[stage_name]["end_time"]
                - self.progress_data[stage_name]["start_time"]
            )

            logger.info(f"Completed stage '{stage_name}' in {duration:.2f}s")

    def get_overall_progress(self) -> dict[str, Any]:
        """
        Get overall progress across all stages.

        Returns:
            Dictionary with overall progress information
        """
        total_elapsed = time.monotonic() - self.start_time if self.start_time else 0

        return {
            "current_stage": self.current_stage,
            "total_elapsed_time": total_elapsed,
            "stages": self.progress_data,
            "overall_status": "running"
            if self.current_stage != "completed"
            else "completed",
        }

    async def report_progress(
        self, callback: Callable[[int, int, str], None] | None
    ) -> None:
        """
        Report progress using the provided callback.

        Args:
            callback: Progress callback function
        """
        if callback and self.current_stage in self.progress_data:
            stage_data = self.progress_data[self.current_stage]
            callback(
                stage_data["completed_items"],
                stage_data["total_items"],
                stage_data.get("message", ""),
            )


class WorkflowManager:
    """Manages complex RAG workflow execution."""

    def __init__(self, pipeline: ProcessingPipeline | None = None) -> None:
        self.pipeline = pipeline if pipeline is not None else ProcessingPipeline()
        self.progress_tracker = ProgressTracker()

    async def execute_full_pipeline(
        self,
        crawl_result: CrawlResult,
        config: dict[str, Any],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the complete RAG processing pipeline.

        Args:
            crawl_result: Crawl result to process
            config: Processing configuration
            progress_callback: Optional progress callback

        Returns:
            Processing result dictionary
        """
        self.progress_tracker.start_stage("initialization", 1)

        # Extract configuration
        deduplication = config.get("deduplication", settings.deduplication_enabled)
        force_update = config.get("force_update", False)

        try:
            # Execute processing pipeline
            result = await self.pipeline.process_crawl_result(
                crawl_result=crawl_result,
                deduplication=deduplication,
                force_update=force_update,
                progress_callback=progress_callback,
            )

            self.progress_tracker.complete_stage("initialization")
            return result

        except Exception as e:
            logger.error(f"Full pipeline execution failed: {e}")
            raise

    async def execute_incremental_update(
        self, source_url: str, new_pages: list[PageContent]
    ) -> dict[str, Any]:
        """
        Execute incremental update for existing source.

        Args:
            source_url: Source URL to update
            new_pages: New pages to process

        Returns:
            Update result dictionary
        """
        self.progress_tracker.start_stage("incremental_update", len(new_pages))

        # Create a mini crawl result for processing
        from ...models.crawl import CrawlResult, CrawlStatistics, CrawlStatus

        mini_crawl_result = CrawlResult(
            request_id="mini-crawl",
            status=CrawlStatus.COMPLETED,
            urls=[],
            pages=new_pages,
            statistics=CrawlStatistics(
                total_pages_crawled=len(new_pages),
                crawl_duration_seconds=0.0,
            ),
        )

        # Process with deduplication enabled
        result = await self.pipeline.process_crawl_result(
            crawl_result=mini_crawl_result,
            deduplication=True,
            force_update=False,
        )

        self.progress_tracker.complete_stage("incremental_update")

        # Calculate actual processing duration
        stage_data = self.progress_tracker.progress_data.get("incremental_update", {})
        start_time = stage_data.get("start_time", 0)
        end_time = stage_data.get("end_time", time.monotonic())
        actual_duration = end_time - start_time

        # Update result with actual processing time
        if (
            isinstance(result, dict)
            and "statistics" in result
            and isinstance(result["statistics"], dict)
        ):
            result["statistics"]["processing_duration_seconds"] = actual_duration

        return result

    async def execute_reprocessing(
        self, source_url: str, force_rechunk: bool = False
    ) -> dict[str, Any]:
        """
        Execute reprocessing of existing source.

        Args:
            source_url: Source URL to reprocess
            force_rechunk: Whether to force re-chunking

        Returns:
            Reprocessing result dictionary
        """
        self.progress_tracker.start_stage("reprocessing", 1)

        # Implementation would retrieve existing pages and reprocess
        # This is a placeholder for the reprocessing logic
        result = {
            "source_url": source_url,
            "force_rechunk": force_rechunk,
            "status": "reprocessed",
        }

        self.progress_tracker.complete_stage("reprocessing")
        return result

    def create_processing_plan(
        self, crawl_result: CrawlResult, config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create a processing plan for the crawl result.

        Args:
            crawl_result: Crawl result to plan for
            config: Processing configuration

        Returns:
            Processing plan dictionary
        """
        total_pages = len(crawl_result.pages)
        estimated_chunks = total_pages * 3  # Estimate 3 chunks per page

        return {
            "total_pages": total_pages,
            "estimated_chunks": estimated_chunks,
            "deduplication_enabled": config.get("deduplication", True),
            "force_update": config.get("force_update", False),
            "estimated_processing_time": total_pages * 0.5,  # 0.5s per page estimate
            "stages": [
                "content_chunking",
                "deduplication_check",
                "embedding_generation",
                "vector_storage",
            ],
        }

    async def monitor_workflow_health(self) -> dict[str, Any]:
        """
        Monitor the health of the workflow system.

        Returns:
            Workflow health status
        """
        pipeline_health = await self.pipeline.get_health_status()
        progress_status = self.progress_tracker.get_overall_progress()

        return {
            "pipeline_health": pipeline_health,
            "progress_status": progress_status,
            "timestamp": datetime.utcnow().isoformat(),
        }


class ProcessingPipeline:
    """Main RAG processing pipeline coordinator."""

    def __init__(self) -> None:
        self.chunker = AdaptiveChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.embedding_pipeline = EmbeddingPipeline()
        self.deduplication_manager = (
            VectorDeduplicationManager()
        )  # Uses default 0.95 threshold
        self.vector_service: VectorService | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the processing pipeline."""
        if not self._initialized:
            self.vector_service = VectorService()
            if self.vector_service is not None:
                await self.vector_service.__aenter__()
            # Pass the vector service to the deduplication manager
            self.deduplication_manager.vector_service = self.vector_service
            await self.embedding_pipeline.initialize()
            self._initialized = True
            logger.info("Processing pipeline initialized")

    async def shutdown(self) -> None:
        """Shutdown the processing pipeline."""
        if self._initialized:
            await self.embedding_pipeline.shutdown()
            if self.vector_service:
                await self.vector_service.__aexit__(None, None, None)  # type: ignore[arg-type]
            self._initialized = False
            logger.info("Processing pipeline shutdown")

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
        Process a crawl result by chunking content and generating embeddings.

        With deduplication enabled, this method:
        1. Queries existing chunks for the source URL
        2. Generates deterministic IDs based on URL and chunk position
        3. Compares content hashes to detect changes
        4. Only upserts new or modified chunks
        5. Deletes orphaned chunks that no longer exist

        Args:
            crawl_result: Result from crawler service
            deduplication: Enable deduplication (defaults to settings.deduplication_enabled)
            force_update: Force update all chunks even if content unchanged
            progress_callback: Optional progress callback

        Returns:
            Dictionary with processing statistics including deduplication metrics
        """
        if not self._initialized:
            await self.initialize()

        # Use settings default if not specified
        if deduplication is None:
            deduplication = settings.deduplication_enabled

        if not crawl_result.pages:
            return {
                "documents_processed": 0,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "chunks_skipped": 0,
                "chunks_updated": 0,
                "chunks_deleted": 0,
            }

        total_pages = len(crawl_result.pages)
        total_chunks = 0
        total_embeddings = 0
        document_chunks = []

        # Deduplication tracking
        chunks_skipped = 0
        chunks_updated = 0
        chunks_deleted = 0
        existing_chunks_map = {}
        legacy_chunks_to_delete = []
        new_chunk_ids = set()  # Track new chunk IDs for orphan detection

        logger.info(
            f"Processing {total_pages} pages for RAG indexing (dedup={deduplication})"
        )

        # Initialize backwards compatibility variables
        use_backwards_compatibility = False
        existing_chunks_list = None

        # Step 1: Get existing chunks if deduplication is enabled
        if deduplication and total_pages > 0:
            source_url = crawl_result.pages[0].url
            if progress_callback:
                progress_callback(
                    0, total_pages + 2, "Retrieving existing chunks for deduplication"
                )

            try:
                existing_chunks = []
                if self.vector_service is not None:
                    existing_chunks = await self.vector_service.get_chunks_by_source(
                        source_url
                    )
                existing_chunks_map = {
                    chunk["id"]: chunk.get("content_hash", "")
                    for chunk in existing_chunks
                }
                logger.info(
                    f"Found {len(existing_chunks_map)} existing chunks for {source_url}"
                )

                # Check if backwards compatibility is needed
                use_backwards_compatibility = (
                    self.deduplication_manager.should_use_backwards_compatibility(
                        existing_chunks_map
                    )
                )
                if use_backwards_compatibility:
                    logger.info(
                        f"Detected {len(existing_chunks_map)} chunks with random UUIDs, enabling backwards compatibility mode"
                    )

                existing_chunks_list = (
                    existing_chunks if use_backwards_compatibility else None
                )

                # Fast path optimization
                if not existing_chunks_map and not force_update:
                    logger.info(
                        "Fast path: No existing chunks found, disabling deduplication for this crawl"
                    )
                    deduplication = False

            except Exception as e:
                logger.warning(
                    f"Could not retrieve existing chunks for deduplication: {e}"
                )
                deduplication = False

        # Process each page
        for i, page in enumerate(crawl_result.pages):
            try:
                if progress_callback:
                    progress_callback(
                        i + 1,
                        total_pages + 2,
                        f"Processing page {i + 1}/{total_pages}: {page.url}",
                    )

                # Process single page
                page_chunks = await self.process_single_page(page, i)

                # Process each chunk
                for sub_chunk_idx, chunk_data in enumerate(page_chunks):
                    composite_chunk_index = f"{i}_{sub_chunk_idx}"

                    # Generate deterministic ID if deduplication is enabled
                    if deduplication:
                        chunk_id = self.deduplication_manager.generate_deterministic_id(
                            page.url, composite_chunk_index
                        )
                        content_hash = self.deduplication_manager.generate_content_hash(
                            chunk_data["text"]
                        )
                        # Track chunk ID for orphan detection
                        new_chunk_ids.add(chunk_id)
                    else:
                        import uuid

                        chunk_id = f"{uuid.uuid4()}"
                        content_hash = None

                    # Check if we should skip this chunk
                    should_skip = False
                    legacy_chunk_to_replace = None

                    if deduplication and not force_update:
                        # Check deterministic ID first
                        if (
                            chunk_id in existing_chunks_map
                            and existing_chunks_map[chunk_id] == content_hash
                        ):
                            chunks_skipped += 1
                            should_skip = True

                        # Backwards compatibility check
                        elif use_backwards_compatibility and existing_chunks_list:
                            legacy_chunk = (
                                self.deduplication_manager.find_legacy_chunk_by_content(
                                    existing_chunks_list, chunk_data["text"]
                                )
                            )
                            if legacy_chunk:
                                legacy_chunk_hash = legacy_chunk.get("content_hash", "")
                                if legacy_chunk_hash == content_hash:
                                    chunks_skipped += 1
                                    legacy_chunk_to_replace = legacy_chunk
                                    legacy_chunks_to_delete.append(legacy_chunk["id"])
                                    should_skip = True

                    if should_skip:
                        continue

                    # Determine if this is an update
                    if deduplication:
                        if chunk_id in existing_chunks_map:
                            chunks_updated += 1
                        elif use_backwards_compatibility and legacy_chunk_to_replace:
                            chunks_updated += 1
                            legacy_chunks_to_delete.append(
                                legacy_chunk_to_replace["id"]
                            )

                    # Create document chunk
                    now = datetime.utcnow()
                    # Determine discovery method based on URL relationship to seed
                    discovery_method = "manual"  # Default fallback
                    if seed_url:
                        if page.url == seed_url:
                            discovery_method = "seed"
                        elif "sitemap" in page.metadata.get("source", "").lower():
                            discovery_method = "sitemap"
                        else:
                            discovery_method = "link_following"

                    doc_chunk = DocumentChunk(
                        id=chunk_id,
                        content=chunk_data["text"],
                        source_url=page.url,
                        source_title=page.title,
                        chunk_index=chunk_data["chunk_index"],
                        word_count=chunk_data["word_count"],
                        char_count=chunk_data["char_count"],
                        metadata={
                            **page.metadata,
                            "sub_chunk_index": sub_chunk_idx,
                            "page_index": i,
                            "start_pos": chunk_data["start_pos"],
                            "end_pos": chunk_data["end_pos"],
                        },
                        content_hash=content_hash,
                        last_modified=now,
                        # Crawl provenance tracking
                        seed_url=seed_url,
                        crawl_session_id=crawl_session_id,
                        discovery_method=discovery_method,
                    )
                    document_chunks.append(doc_chunk)
                    total_chunks += 1

            except Exception as e:
                logger.error(f"Error processing page {page.url}: {e}")
                continue

        if not document_chunks and not deduplication:
            logger.warning("No document chunks created from crawl result")
            return {
                "documents_processed": 0,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "chunks_skipped": 0,
                "chunks_updated": 0,
                "chunks_deleted": 0,
            }

        # Step 2: Handle orphaned chunks
        if deduplication and settings.delete_orphaned_chunks and total_pages > 0:
            # Identify orphaned chunks (existing chunks that are no longer present)
            orphaned_ids = set(existing_chunks_map.keys()) - new_chunk_ids
            all_ids_to_delete = orphaned_ids.union(set(legacy_chunks_to_delete))

            if all_ids_to_delete:
                if progress_callback:
                    progress_callback(
                        total_pages + 1,
                        total_pages + 3,
                        f"Deleting {len(orphaned_ids)} orphaned chunks and {len(legacy_chunks_to_delete)} legacy chunks",
                    )

                try:
                    chunks_deleted = 0
                    if self.vector_service is not None:
                        chunks_deleted = await self.vector_service.delete_chunks_by_ids(
                            list(all_ids_to_delete)
                        )
                    logger.info(
                        f"Deleted {chunks_deleted} chunks ({len(orphaned_ids)} orphaned, {len(legacy_chunks_to_delete)} legacy)"
                    )
                except Exception as e:
                    logger.error(f"Error deleting chunks: {e}")

        # If all chunks were skipped, return early
        if not document_chunks:
            logger.info(
                f"No chunks to process (skipped={chunks_skipped}, deleted={chunks_deleted})"
            )
            return {
                "documents_processed": total_pages,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "chunks_skipped": chunks_skipped,
                "chunks_updated": chunks_updated,
                "chunks_deleted": chunks_deleted,
            }

        # Generate embeddings with parallel pipeline processing
        if progress_callback:
            progress_callback(
                total_pages,
                total_pages + 1,
                f"Generating embeddings for {len(document_chunks)} chunks",
            )

        try:
            embedding_start_time = time.monotonic()

            # Process chunks with embeddings
            await self.embedding_pipeline.process_chunks_parallel(
                document_chunks, progress_callback, total_pages
            )
            total_embeddings = len(document_chunks)

            embedding_duration = time.monotonic() - embedding_start_time
            logger.info(
                f"Generated {total_embeddings} embeddings in {embedding_duration:.2f}s - "
                f"{total_embeddings / embedding_duration:.1f} embeddings/sec"
            )

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

        # Store in vector database
        if progress_callback:
            progress_callback(
                total_pages + 1,
                total_pages + 2,
                f"Storing {len(document_chunks)} embeddings in vector database",
            )

        try:
            storage_start_time = time.monotonic()
            stored_count = 0
            if self.vector_service is not None:
                stored_count = await self.vector_service.upsert_documents(
                    document_chunks
                )
            storage_duration = time.monotonic() - storage_start_time
            logger.info(
                f"Stored {stored_count} document chunks in vector database in {storage_duration:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise

        return {
            "documents_processed": total_pages,
            "chunks_created": total_chunks,
            "embeddings_generated": total_embeddings,
            "chunks_stored": stored_count,
            "chunks_skipped": chunks_skipped,
            "chunks_updated": chunks_updated,
            "chunks_deleted": chunks_deleted,
        }

    async def process_pages_batch(
        self, pages: list[PageContent], batch_size: int = 50
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Process a batch of pages.

        Args:
            pages: List of pages to process
            batch_size: Size of processing batches

        Returns:
            Tuple of (processed_chunks, statistics)
        """
        all_chunks = []
        stats = {"pages_processed": 0, "chunks_created": 0, "errors": 0}

        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]

            for page_idx, page in enumerate(batch):
                try:
                    page_chunks = await self.process_single_page(page, i + page_idx)
                    all_chunks.extend(page_chunks)
                    stats["pages_processed"] += 1
                    stats["chunks_created"] += len(page_chunks)
                except Exception as e:
                    logger.error(f"Error processing page {page.url}: {e}")
                    stats["errors"] += 1

        return all_chunks, stats

    async def process_single_page(
        self, page: PageContent, page_index: int = 0
    ) -> list[dict[str, Any]]:
        """
        Process a single page to create chunks.

        Args:
            page: Page content to process
            page_index: Index of the page in the crawl

        Returns:
            List of chunk dictionaries
        """
        # Create metadata for chunks
        metadata = {
            "source_url": page.url,
            "source_title": page.title,
            "page_index": page_index,
            "markdown": page.markdown,
            "html": page.html,
            "links": page.links,
            "images": page.images,
            "page_word_count": page.word_count,
            "timestamp": page.timestamp.isoformat() if page.timestamp else None,
            **page.metadata,
        }

        # Chunk the page content
        chunks = self.chunker.chunk_text(page.content, metadata)

        return chunks

    async def store_chunks_batch(
        self, chunks: list[DocumentChunk], batch_size: int = 100
    ) -> int:
        """
        Store chunks in the vector database in batches.

        Args:
            chunks: Chunks to store
            batch_size: Size of storage batches

        Returns:
            Number of chunks stored
        """
        total_stored = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                stored = 0
                if self.vector_service is not None:
                    stored = await self.vector_service.upsert_documents(batch)
                total_stored += stored
            except Exception as e:
                logger.error(f"Error storing chunk batch: {e}")

        return total_stored

    def calculate_processing_statistics(
        self,
        total_pages: int,
        total_chunks: int,
        processing_time: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Calculate comprehensive processing statistics.

        Args:
            total_pages: Total pages processed
            total_chunks: Total chunks created
            processing_time: Total processing time
            **kwargs: Additional statistics

        Returns:
            Statistics dictionary
        """
        pages_per_second = total_pages / processing_time if processing_time > 0 else 0
        chunks_per_second = total_chunks / processing_time if processing_time > 0 else 0

        stats = {
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "processing_time": processing_time,
            "pages_per_second": pages_per_second,
            "chunks_per_second": chunks_per_second,
            "avg_chunks_per_page": total_chunks / total_pages if total_pages > 0 else 0,
            **kwargs,
        }

        return stats

    async def validate_processing_result(
        self, result: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate the processing result for quality and completeness.

        Args:
            result: Processing result to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check for required fields
        required_fields = [
            "documents_processed",
            "chunks_created",
            "embeddings_generated",
        ]

        for field in required_fields:
            if field not in result:
                issues.append(f"Missing required field: {field}")

        # Check for logical consistency
        if result.get("chunks_created", 0) != result.get("embeddings_generated", 0):
            issues.append("Chunk count doesn't match embedding count")

        # Check for reasonable ratios
        docs_processed = result.get("documents_processed", 0)
        chunks_created = result.get("chunks_created", 0)

        if docs_processed > 0:
            chunks_per_doc = chunks_created / docs_processed
            if chunks_per_doc > 20:  # Very high ratio
                issues.append(
                    f"Unusually high chunks per document: {chunks_per_doc:.1f}"
                )
            elif chunks_per_doc < 0.5:  # Very low ratio
                issues.append(
                    f"Unusually low chunks per document: {chunks_per_doc:.1f}"
                )

        return len(issues) == 0, issues

    async def get_health_status(self) -> dict[str, Any]:
        """
        Get health status of the processing pipeline.

        Returns:
            Health status dictionary
        """
        health_status = {
            "pipeline_initialized": self._initialized,
            "chunker_type": type(self.chunker).__name__,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check embedding pipeline health
        if self.embedding_pipeline:
            embedding_health = await self.embedding_pipeline.health_check()
            health_status["embedding_pipeline"] = embedding_health

        # Check vector service health
        if self.vector_service:
            vector_healthy = await self.vector_service.health_check()
            health_status["vector_service"] = (
                "healthy" if vector_healthy else "unhealthy"
            )

        return health_status
