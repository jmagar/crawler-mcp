"""
Streaming utilities for processing large datasets efficiently.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Generic, TypeVar

from ..config import settings
from ..models.rag import DocumentChunk

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ChunkStream:
    """
    Async stream processor for document chunks with backpressure control.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        max_buffer_size: int = 1000,
    ):
        """
        Initialize chunk stream.

        Args:
            batch_size: Size of batches to process
            max_buffer_size: Maximum items to buffer
        """
        self.batch_size = batch_size or settings.default_batch_size
        self.max_buffer_size = max_buffer_size

        # Queues for streaming
        self.input_queue: asyncio.Queue[DocumentChunk | None] = asyncio.Queue(
            maxsize=max_buffer_size
        )
        self.output_queue: asyncio.Queue[DocumentChunk | None] = asyncio.Queue(
            maxsize=max_buffer_size
        )

        # Statistics
        self.total_processed = 0
        self.total_errors = 0

        # Control
        self._processing = False
        self._processor_task: asyncio.Task[None] | None = None

    async def start_processing(
        self,
        processor: Callable[[list[DocumentChunk]], AsyncIterator[DocumentChunk]],
    ) -> None:
        """
        Start processing chunks in batches.

        Args:
            processor: Async function to process batches
        """
        if self._processing:
            return

        self._processing = True
        self._processor_task = asyncio.create_task(self._process_batches(processor))
        logger.info(f"Started chunk stream processing (batch_size: {self.batch_size})")

    async def _process_batches(
        self,
        processor: Callable[[list[DocumentChunk]], AsyncIterator[DocumentChunk]],
    ) -> None:
        """Process chunks in batches."""
        batch: list[DocumentChunk] = []

        while self._processing:
            try:
                # Get chunk with timeout to allow periodic batch processing
                chunk = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)

                if chunk is None:
                    # Sentinel value - process remaining batch and stop
                    if batch:
                        await self._process_batch(batch, processor)
                    await self.output_queue.put(None)
                    break

                batch.append(chunk)

                # Process batch when full
                if len(batch) >= self.batch_size:
                    await self._process_batch(batch, processor)
                    batch = []

            except TimeoutError:
                # Timeout - process partial batch if exists
                if batch:
                    await self._process_batch(batch, processor)
                    batch = []

            except Exception as e:
                logger.error(f"Error in stream processing: {e}")
                self.total_errors += 1

    async def _process_batch(
        self,
        batch: list[DocumentChunk],
        processor: Callable[[list[DocumentChunk]], AsyncIterator[DocumentChunk]],
    ) -> None:
        """Process a single batch."""
        try:
            # Process batch and stream results
            async for processed_chunk in processor(batch):
                await self.output_queue.put(processed_chunk)
                self.total_processed += 1

            logger.debug(f"Processed batch of {len(batch)} chunks")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.total_errors += 1

            # Put unprocessed chunks back for retry
            for chunk in batch:
                chunk.metadata["processing_error"] = str(e)
                await self.output_queue.put(chunk)

    async def add_chunk(self, chunk: DocumentChunk) -> None:
        """
        Add a chunk to the processing stream.

        Args:
            chunk: Document chunk to process
        """
        await self.input_queue.put(chunk)

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Add multiple chunks to the processing stream.

        Args:
            chunks: List of document chunks
        """
        for chunk in chunks:
            await self.add_chunk(chunk)

    async def finish_input(self) -> None:
        """Signal that no more input will be added."""
        await self.input_queue.put(None)

    async def get_processed(self) -> DocumentChunk | None:
        """
        Get next processed chunk.

        Returns:
            Processed chunk or None if stream ended
        """
        return await self.output_queue.get()

    async def get_all_processed(self) -> list[DocumentChunk]:
        """
        Get all processed chunks (blocks until processing complete).

        Returns:
            List of all processed chunks
        """
        results = []

        while True:
            chunk = await self.get_processed()
            if chunk is None:
                break
            results.append(chunk)

        return results

    async def stop(self) -> None:
        """Stop processing."""
        self._processing = False

        if self._processor_task:
            self._processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processor_task

    def get_stats(self) -> dict[str, Any]:
        """Get stream statistics."""
        return {
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "batch_size": self.batch_size,
        }


class AsyncBatchIterator(Generic[T]):
    """
    Async iterator that yields items in batches.
    """

    def __init__(
        self,
        items: list[T] | AsyncIterator[T],
        batch_size: int | None = None,
    ):
        """
        Initialize batch iterator.

        Args:
            items: List of items or async iterator
            batch_size: Size of batches to yield
        """
        self.items = items
        self.batch_size = batch_size or settings.default_batch_size
        self._index = 0

    def __aiter__(self) -> "AsyncBatchIterator[T]":
        """Return self as async iterator."""
        return self

    async def __anext__(self) -> list[T]:
        """Get next batch."""
        if isinstance(self.items, list):
            # List-based iteration
            if self._index >= len(self.items):
                raise StopAsyncIteration

            batch = self.items[self._index : self._index + self.batch_size]
            self._index += self.batch_size
            return batch

        else:
            # Async iterator-based iteration
            batch = []

            try:
                for _ in range(self.batch_size):
                    item = await self.items.__anext__()
                    batch.append(item)
            except StopAsyncIteration:
                if not batch:
                    raise

            return batch


async def stream_process_chunks(
    chunks: list[DocumentChunk] | AsyncIterator[DocumentChunk],
    embed_func: Callable[[list[str]], Awaitable[list[list[float]]]],
    store_func: Callable[[list[DocumentChunk]], Awaitable[int]],
    batch_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    ctx: Any | None = None,
) -> tuple[int, int]:
    """
    Stream process chunks with embedding and storage.

    Args:
        chunks: Chunks to process (list or async iterator)
        embed_func: Function to generate embeddings
        store_func: Function to store chunks
        batch_size: Batch size for processing
        progress_callback: Optional progress callback
        ctx: Optional context for client-facing progress messages

    Returns:
        Tuple of (total processed, total errors)
    """
    batch_size = batch_size or settings.default_batch_size
    total_processed = 0
    total_errors = 0

    # Create batch iterator
    if isinstance(chunks, list):
        total_chunks = len(chunks)
        batch_iterator = AsyncBatchIterator(chunks, batch_size)
    else:
        total_chunks = None  # Unknown for iterators
        batch_iterator = AsyncBatchIterator(chunks, batch_size)

    # Process batches
    batch_num = 0
    async for batch in batch_iterator:
        batch_num += 1

        try:
            # Extract texts for embedding
            texts = [chunk.content for chunk in batch]

            # Generate embeddings
            embeddings = await embed_func(texts)

            # Validate embedding/batch length matching
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise ValueError(
                    f"Embedding function returned {len(embeddings) if hasattr(embeddings, '__len__') else 'non-sequence'} "
                    f"embeddings for {len(batch)} chunks"
                )

            # Attach embeddings to chunks
            for chunk, embedding in zip(batch, embeddings, strict=True):
                chunk.embedding = embedding

            # Store chunks
            stored = await store_func(batch)
            total_processed += stored

            # Report progress
            if progress_callback and total_chunks:
                progress_callback(total_processed, total_chunks)

            # Client-facing progress
            if ctx and total_chunks:
                await ctx.info(f"Processed {total_processed}/{total_chunks} chunks")

            logger.debug(
                f"Processed batch {batch_num} ({len(batch)} chunks, {stored} stored)"
            )

        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            total_errors += len(batch)

    logger.info(
        f"Stream processing complete: {total_processed} processed, {total_errors} errors"
    )

    return total_processed, total_errors


async def parallel_stream_process(
    chunks: list[DocumentChunk],
    processors: list[Callable[[DocumentChunk], DocumentChunk]],
    max_workers: int = 4,
) -> list[DocumentChunk]:
    """
    Process chunks in parallel with multiple processors.

    Args:
        chunks: Chunks to process
        processors: List of processor functions
        max_workers: Maximum parallel workers

    Returns:
        Processed chunks
    """
    semaphore = asyncio.Semaphore(max_workers)

    async def process_chunk(chunk: DocumentChunk) -> DocumentChunk:
        """Process single chunk through all processors."""
        async with semaphore:
            result = chunk
            for processor in processors:
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(result)
                else:
                    result = processor(result)
            return result

    # Process all chunks in parallel
    tasks = [process_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out errors
    processed: list[DocumentChunk] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Chunk processing error: {result}")
        elif result is not None:
            processed.append(result)  # type: ignore[arg-type]

    return processed
