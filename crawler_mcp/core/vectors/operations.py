"""
Document CRUD operations for Qdrant vector database.
"""

import logging
from typing import Any, TypedDict

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    UpdateStatus,
)

from ...models.rag import DocumentChunk
from .base import BaseVectorService, _parse_timestamp

logger = logging.getLogger(__name__)


class BulkOperation(TypedDict, total=False):
    """Type definition for bulk operations."""

    action: str  # Required: "update", "delete", "insert"
    id: str  # Required: document ID
    vector: list[float]  # Optional: for insert/update operations
    payload: dict[str, Any]  # Optional: for insert/update operations


class DocumentOperations(BaseVectorService):
    """
    Handles all document CRUD operations in the vector database.
    """

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        """
        Initialize the document operations manager.

        Args:
            client: Optional shared Qdrant client instance
        """
        super().__init__(client)

    async def upsert_documents(
        self, documents: list[DocumentChunk], batch_size: int = 100
    ) -> int:
        """
        Upsert document chunks into the vector database.

        Args:
            documents: List of document chunks to upsert
            batch_size: Size of batches for bulk operations

        Returns:
            Number of documents successfully upserted
        """
        if not documents:
            return 0

        # Ensure collection exists (delegate to collections module)
        from .collections import CollectionManager

        collection_manager = CollectionManager(self.client)
        await collection_manager.ensure_collection_exists()

        total_batches = (len(documents) + batch_size - 1) // batch_size
        total_upserted = 0

        logger.info(f"Upserting {len(documents)} documents in {total_batches} batches")

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(documents))
            batch = documents[start_idx:end_idx]

            logger.debug(
                f"Upserting batch {batch_idx + 1}/{total_batches} ({len(batch)} documents)"
            )

            try:
                points = []
                for doc in batch:
                    if not doc.embedding:
                        logger.warning(f"Skipping document {doc.id} - no embedding")
                        continue

                    # Validate embedding dimensionality
                    if len(doc.embedding) != self.vector_size:
                        logger.warning(
                            f"Skipping document {doc.id} - embedding dimension mismatch: "
                            f"got {len(doc.embedding)}, expected {self.vector_size}"
                        )
                        continue

                    # Prepare point data
                    point = PointStruct(
                        id=doc.id,
                        vector=doc.embedding,
                        payload={
                            "content": doc.content,
                            "source_url": doc.source_url,
                            "source_title": doc.source_title,
                            "chunk_index": doc.chunk_index,
                            "word_count": doc.word_count,
                            "char_count": doc.char_count,
                            "timestamp": doc.timestamp.isoformat(),
                            # Crawl provenance tracking
                            "seed_url": doc.seed_url,
                            "crawl_session_id": doc.crawl_session_id,
                            "discovery_method": doc.discovery_method,
                            **doc.metadata,
                        },
                    )
                    points.append(point)

                if points:
                    # Upsert the batch with client recreation on error
                    try:
                        client = await self._get_client()
                        result = await client.upsert(
                            collection_name=self.collection_name, points=points
                        )
                    except Exception as e:
                        if await self._handle_client_error(e):
                            client = await self._get_client()
                            result = await client.upsert(
                                collection_name=self.collection_name, points=points
                            )
                        else:
                            raise

                    if result.status == UpdateStatus.COMPLETED:
                        total_upserted += len(points)
                        logger.debug(f"Successfully upserted batch {batch_idx + 1}")
                    else:
                        logger.error(
                            f"Failed to upsert batch {batch_idx + 1}: {result.status}"
                        )

            except Exception as e:
                logger.error(f"Error upserting batch {batch_idx + 1}: {e}")
                # Continue with next batch rather than failing completely

        logger.info(f"Total documents upserted: {total_upserted}")
        return total_upserted

    async def get_document_by_id(self, document_id: str) -> DocumentChunk | None:
        """
        Retrieve a specific document by ID.

        Args:
            document_id: The document ID to retrieve

        Returns:
            DocumentChunk if found, None otherwise
        """
        # Ensure collection exists
        from .collections import CollectionManager

        collection_manager = CollectionManager(self.client)
        await collection_manager.ensure_collection_exists()

        try:
            client = await self._get_client()
            result = await client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id],
                with_payload=True,
                with_vectors=False,
            )

            if not result:
                return None

            point = result[0]
            payload = getattr(point, "payload", None)
            if payload is None:
                return None

            return DocumentChunk(
                id=str(getattr(point, "id", "unknown")),
                content=payload.get("content", ""),
                source_url=payload.get("source_url", ""),
                source_title=payload.get("source_title"),
                chunk_index=payload.get("chunk_index", 0),
                word_count=payload.get("word_count", 0),
                char_count=payload.get("char_count", 0),
                timestamp=_parse_timestamp(payload.get("timestamp", "")),
                # Crawl provenance tracking
                seed_url=payload.get("seed_url"),
                crawl_session_id=payload.get("crawl_session_id"),
                discovery_method=payload.get("discovery_method"),
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
                        "seed_url",
                        "crawl_session_id",
                        "discovery_method",
                    ]
                },
            )

        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            return None

    async def delete_documents_by_source(self, source_url: str) -> int:
        """
        Delete all documents from a specific source.

        Args:
            source_url: URL of the source to delete documents from

        Returns:
            Number of documents deleted
        """
        # Ensure collection exists
        from .collections import CollectionManager

        collection_manager = CollectionManager(self.client)
        await collection_manager.ensure_collection_exists()

        try:
            # Delete points with matching source_url
            client = await self._get_client()
            result = await client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_url", match=MatchValue(value=source_url)
                        )
                    ]
                ),
            )

            if result.status == UpdateStatus.COMPLETED:
                logger.info(f"Deleted documents from source: {source_url}")
                return 1  # Qdrant doesn't return exact count for filter deletes
            else:
                logger.error(
                    f"Failed to delete documents from source {source_url}: {result.status}"
                )
                return 0

        except Exception as e:
            logger.error(f"Error deleting documents from source {source_url}: {e}")
            return 0

    async def get_chunks_by_source(self, source_url: str) -> list[dict[str, Any]]:
        """
        Get all existing chunks for a source URL.

        Args:
            source_url: URL of the source to retrieve chunks from

        Returns:
            List of chunk documents with their IDs and content hashes
        """
        # Ensure collection exists
        from .collections import CollectionManager

        collection_manager = CollectionManager(self.client)
        await collection_manager.ensure_collection_exists()

        try:
            # Query all points with matching source_url
            client = await self._get_client()
            response = await client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_url", match=MatchValue(value=source_url)
                        )
                    ]
                ),
                limit=10000,  # Large limit to get all chunks
                with_payload=True,
                with_vectors=False,  # Don't need embeddings for dedup
            )

            chunks = []
            for point in response[0]:  # response is (points, next_offset)
                payload = point.payload or {}
                chunk_data = {
                    "id": point.id,
                    "content": payload.get("content", ""),
                    "content_hash": payload.get("content_hash"),
                    "chunk_index": payload.get("chunk_index", 0),
                    "metadata": payload.get("metadata", {}),
                }
                chunks.append(chunk_data)

            logger.info(f"Retrieved {len(chunks)} chunks for source: {source_url}")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks for source {source_url}: {e}")
            return []

    async def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
        """
        Delete specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks deleted
        """
        if not chunk_ids:
            return 0

        # Ensure collection exists
        from .collections import CollectionManager

        collection_manager = CollectionManager(self.client)
        await collection_manager.ensure_collection_exists()

        try:
            # Delete points by IDs
            client = await self._get_client()
            result = await client.delete(
                collection_name=self.collection_name,
                points_selector=chunk_ids,  # Direct ID list
            )

            if result.status == UpdateStatus.COMPLETED:
                logger.info(f"Deleted {len(chunk_ids)} chunks by ID")
                return len(chunk_ids)
            else:
                logger.error(f"Failed to delete chunks by ID: {result.status}")
                return 0

        except Exception as e:
            logger.error(f"Error deleting chunks by ID: {e}")
            return 0

    async def bulk_update_documents(self, operations: list[BulkOperation]) -> int:
        """
        Perform bulk update operations on documents.

        Args:
            operations: List of typed bulk operation dictionaries

        Returns:
            Number of successful operations
        """
        # Placeholder for future bulk operations implementation
        # This would handle mixed operations like update/delete/insert
        logger.warning("Bulk update operations not yet implemented")
        return 0
