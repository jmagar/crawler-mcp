"""
Content deduplication and hash management for RAG operations.

This module provides comprehensive deduplication capabilities including
content hashing, similarity detection, and backwards compatibility handling.
"""

import hashlib
import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any

from ...models.rag import DocumentChunk
from ..utils import normalize_url

logger = logging.getLogger(__name__)


class ContentHasher:
    """Content hashing utilities for deduplication."""

    @staticmethod
    def hash_content(content: str) -> str:
        """
        Generate SHA256 hash of normalized content.

        Args:
            content: Text content to hash

        Returns:
            SHA256 hash hexdigest string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_chunk_metadata(chunk: DocumentChunk) -> str:
        """
        Generate hash including content and key metadata.

        Args:
            chunk: Document chunk to hash

        Returns:
            Combined hash of content and metadata
        """
        # Include key metadata that affects uniqueness
        metadata_string = f"{chunk.source_url}:{chunk.source_title}:{chunk.chunk_index}"
        combined_string = f"{chunk.content}:{metadata_string}"
        return hashlib.sha256(combined_string.encode("utf-8")).hexdigest()

    @staticmethod
    def normalize_whitespace(content: str) -> str:
        """
        Normalize whitespace for consistent hashing.

        Args:
            content: Content to normalize

        Returns:
            Content with normalized whitespace
        """
        # Replace multiple whitespace with single space
        normalized = re.sub(r"\s+", " ", content.strip())
        return normalized

    @staticmethod
    def extract_text_features(content: str) -> dict[str, Any]:
        """
        Extract features for similarity comparison.

        Args:
            content: Content to analyze

        Returns:
            Dictionary of extracted features
        """
        words = content.lower().split()
        unique_words = set(words)

        return {
            "word_count": len(words),
            "unique_word_count": len(unique_words),
            "char_count": len(content),
            "avg_word_length": sum(len(word) for word in words) / len(words)
            if words
            else 0,
            "vocabulary_ratio": len(unique_words) / len(words) if words else 0,
            "first_50_chars": content[:50],
            "last_50_chars": content[-50:] if len(content) > 50 else content,
        }


class SimilarityDetector:
    """Advanced similarity detection for near-duplicate content."""

    def __init__(self, min_similarity: float = 0.85):
        self.min_similarity = min_similarity

    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 1.0 if not words1 and not words2 else 0.0

        return len(intersection) / len(union)

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts using simple word vectors.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        # Create vocabulary
        vocab = set(words1 + words2)

        # Create word count vectors
        vec1 = [words1.count(word) for word in vocab]
        vec2 = [words2.count(word) for word in vocab]

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 1.0 if magnitude1 == magnitude2 else 0.0

        return float(dot_product / (magnitude1 * magnitude2))

    def calculate_levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Levenshtein similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Levenshtein similarity score (0-1)
        """
        # Simple Levenshtein distance implementation
        if len(text1) > 1000 or len(text2) > 1000:
            # Use first 1000 chars for performance
            text1 = text1[:1000]
            text2 = text2[:1000]

        if len(text1) < len(text2):
            text1, text2 = text2, text1

        if len(text2) == 0:
            return 0.0

        # Create distance matrix
        previous_row = list(range(len(text2) + 1))
        for i, c1 in enumerate(text1):
            current_row = [i + 1]
            for j, c2 in enumerate(text2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        distance = previous_row[-1]
        max_len = max(len(text1), len(text2))

        return 1.0 - (distance / max_len)

    def detect_near_duplicates(
        self, new_chunks: list[DocumentChunk], existing_chunks: list[dict[str, Any]]
    ) -> list[tuple[DocumentChunk, dict[str, Any], float]]:
        """
        Detect near-duplicate content between new and existing chunks.

        Args:
            new_chunks: New chunks to check
            existing_chunks: Existing chunks to compare against

        Returns:
            List of tuples (new_chunk, existing_chunk, similarity_score)
        """
        near_duplicates = []

        for new_chunk in new_chunks:
            for existing_chunk in existing_chunks:
                existing_content = existing_chunk.get("content", "")

                # Calculate multiple similarity metrics
                jaccard_sim = self.calculate_jaccard_similarity(
                    new_chunk.content, existing_content
                )
                cosine_sim = self.calculate_cosine_similarity(
                    new_chunk.content, existing_content
                )

                # Use weighted average of similarities
                combined_similarity = (jaccard_sim * 0.6) + (cosine_sim * 0.4)

                if combined_similarity >= self.min_similarity:
                    near_duplicates.append(
                        (new_chunk, existing_chunk, combined_similarity)
                    )

        return near_duplicates


class DeduplicationManager(ABC):
    """
    Abstract base class for managing content deduplication across the RAG system.

    This class provides a framework for deduplication operations including content
    hashing, similarity detection, and chunk lifecycle management. Subclasses must
    implement the abstract methods to provide vector database-specific functionality
    for finding, identifying, and cleaning up chunks.

    Abstract methods that must be implemented:
    - find_existing_chunks: Query vector DB for existing chunks by source URL
    - identify_orphaned_chunks: Find chunks that no longer have corresponding content
    - cleanup_orphaned_chunks: Remove orphaned chunks from the vector database

    The class provides concrete implementations for:
    - Content hashing and normalization
    - Similarity detection and deduplication logic
    - Backwards compatibility handling
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.hasher = ContentHasher()
        self.similarity_detector = SimilarityDetector(similarity_threshold)
        self.hash_cache: dict[str, str] = {}
        self.similarity_threshold = similarity_threshold

    async def deduplicate_chunks(
        self,
        chunks: list[DocumentChunk],
        existing_chunks: list[dict[str, Any]] | None = None,
    ) -> tuple[list[DocumentChunk], list[DocumentChunk]]:
        """
        Deduplicate chunks against existing content.

        Args:
            chunks: New chunks to process
            existing_chunks: Existing chunks to compare against

        Returns:
            Tuple of (unique_chunks, duplicate_chunks)
        """
        if not existing_chunks:
            # No existing chunks, all are unique
            return chunks, []

        unique_chunks = []
        duplicate_chunks = []

        # Build hash map of existing content
        existing_hashes = {
            chunk.get("content_hash", ""): chunk
            for chunk in existing_chunks
            if chunk.get("content_hash")
        }

        for chunk in chunks:
            content_hash = self.generate_content_hash(chunk.content)
            chunk.content_hash = content_hash

            # Check for exact hash match
            if content_hash in existing_hashes:
                duplicate_chunks.append(chunk)
                logger.debug(f"Exact duplicate found for chunk {chunk.id}")
                continue

            # Check for near-duplicates using similarity detection
            near_duplicates = self.similarity_detector.detect_near_duplicates(
                [chunk], existing_chunks
            )

            if near_duplicates:
                # Found similar content
                _, existing_chunk, similarity = near_duplicates[0]
                duplicate_chunks.append(chunk)
                logger.debug(
                    f"Near-duplicate found for chunk {chunk.id} with similarity {similarity:.3f}"
                )
            else:
                unique_chunks.append(chunk)

        return unique_chunks, duplicate_chunks

    def generate_content_hash(self, content: str) -> str:
        """
        Generate content hash for deduplication.

        Args:
            content: Content to hash

        Returns:
            Content hash string
        """
        normalized_content = self.hasher.normalize_whitespace(content)
        return self.hasher.hash_content(normalized_content)

    def generate_deterministic_id(self, url: str, chunk_index: str | int) -> str:
        """
        Generate deterministic ID from URL and chunk index.

        Args:
            url: Source URL
            chunk_index: Index of the chunk within the document

        Returns:
            Deterministic UUID string
        """
        normalized_url = normalize_url(url)
        id_string = f"{normalized_url}:{chunk_index}"
        # Generate a deterministic UUID from the hash
        hash_bytes = hashlib.sha256(id_string.encode()).digest()[:16]
        # Create UUID from the first 16 bytes of the hash
        deterministic_uuid = uuid.UUID(bytes=hash_bytes)
        return str(deterministic_uuid)

    def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two content strings.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity score (0-1)
        """
        return self.similarity_detector.calculate_jaccard_similarity(content1, content2)

    @abstractmethod
    async def find_existing_chunks(self, source_url: str) -> list[dict[str, Any]]:
        """
        Find existing chunks for a source URL.

        This method must be implemented by subclasses to query the specific
        vector database and return existing chunks for the given source URL.

        Args:
            source_url: Source URL to find chunks for

        Returns:
            List of existing chunk dictionaries

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement find_existing_chunks")

    @abstractmethod
    async def identify_orphaned_chunks(self, source_url: str) -> list[str]:
        """
        Identify orphaned chunks that no longer have corresponding content.

        This method must be implemented by subclasses to identify chunks that
        exist in the vector database but don't have corresponding content in
        the new crawl data.

        Args:
            source_url: Source URL to check for orphans

        Returns:
            List of orphaned chunk IDs

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement identify_orphaned_chunks")

    @abstractmethod
    async def cleanup_orphaned_chunks(self, chunk_ids: list[str]) -> int:
        """
        Clean up orphaned chunks by deleting them.

        This method must be implemented by subclasses to delete the specified
        chunks from the specific vector database implementation.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks successfully deleted

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement cleanup_orphaned_chunks")

    def normalize_content_for_comparison(self, content: str) -> str:
        """
        Normalize content for consistent comparison.

        Args:
            content: Content to normalize

        Returns:
            Normalized content
        """
        return self.hasher.normalize_whitespace(content)

    def extract_content_fingerprint(self, content: str) -> dict[str, Any]:
        """
        Extract a fingerprint of content for fast comparison.

        Args:
            content: Content to fingerprint

        Returns:
            Content fingerprint dictionary
        """
        return self.hasher.extract_text_features(content)

    async def bulk_deduplication(
        self, source_chunks: dict[str, list[DocumentChunk]]
    ) -> dict[str, tuple[list[DocumentChunk], list[DocumentChunk]]]:
        """
        Perform bulk deduplication across multiple sources.

        Args:
            source_chunks: Dictionary mapping source URLs to their chunks

        Returns:
            Dictionary mapping source URLs to (unique_chunks, duplicate_chunks) tuples
        """
        results = {}

        for source_url, chunks in source_chunks.items():
            existing_chunks = await self.find_existing_chunks(source_url)
            unique_chunks, duplicate_chunks = await self.deduplicate_chunks(
                chunks, existing_chunks
            )
            results[source_url] = (unique_chunks, duplicate_chunks)

        return results

    # Backwards compatibility methods
    def is_random_uuid(self, chunk_id: str) -> bool:
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

    def find_legacy_chunk_by_content(
        self, existing_chunks: list[dict[str, Any]], content: str
    ) -> dict[str, Any] | None:
        """
        Find a legacy chunk by content hash for backwards compatibility.

        In mixed environments, we need to match content even if the chunk
        has a random UUID instead of a deterministic ID.

        Args:
            existing_chunks: List of existing chunks
            content: Content to match against

        Returns:
            Matching chunk or None
        """
        content_hash = self.generate_content_hash(content)

        for chunk in existing_chunks:
            # Direct content hash comparison
            if chunk.get("content_hash") == content_hash:
                return chunk

            # Fallback: direct content comparison for chunks without hashes
            if chunk.get("content") == content:
                return chunk

        return None

    def should_use_backwards_compatibility(
        self, existing_chunks_map: dict[str, str]
    ) -> bool:
        """
        Determine if backwards compatibility mode should be used.

        This checks if the existing chunks contain random UUIDs, indicating
        a mixed environment that needs backwards compatibility handling.

        Args:
            existing_chunks_map: Map of chunk_id -> content_hash

        Returns:
            True if backwards compatibility is needed
        """
        if not existing_chunks_map:
            return False

        # Sample a few chunk IDs to determine if they're random UUIDs
        sample_size = min(5, len(existing_chunks_map))
        sample_ids = list(existing_chunks_map.keys())[:sample_size]

        random_uuid_count = sum(
            1 for chunk_id in sample_ids if self.is_random_uuid(chunk_id)
        )

        # If more than half are random UUIDs, use backwards compatibility
        return random_uuid_count > (sample_size // 2)


class VectorDeduplicationManager(DeduplicationManager):
    """
    Concrete implementation of DeduplicationManager that integrates with VectorService
    to perform deduplication operations against a vector database.
    """

    def __init__(self, vector_service: Any = None, similarity_threshold: float = 0.95):
        """
        Initialize VectorDeduplicationManager.

        Args:
            vector_service: VectorService instance for database operations
            similarity_threshold: Threshold for similarity detection
        """
        super().__init__(similarity_threshold)
        self.vector_service = vector_service

    async def find_existing_chunks(self, source_url: str) -> list[dict[str, Any]]:
        """
        Find existing chunks for a source URL by querying the vector database.

        Args:
            source_url: Source URL to search for

        Returns:
            List of existing chunk dictionaries
        """
        if not self.vector_service:
            return []

        try:
            # Use the vector service to find chunks by source URL
            results = await self.vector_service.operations.get_chunks_by_source(
                source_url
            )

            # Convert to the expected format
            chunks = []
            for result in results:
                chunk_data = {
                    "id": result.id,
                    "content": result.payload.get("content", ""),
                    "content_hash": result.payload.get("content_hash", ""),
                    "source_url": result.payload.get("source_url", ""),
                    "metadata": result.payload.get("metadata", {}),
                }
                chunks.append(chunk_data)

            logger.debug(f"Found {len(chunks)} existing chunks for {source_url}")
            return chunks

        except Exception as e:
            logger.warning(f"Error finding existing chunks for {source_url}: {e}")
            return []

    async def identify_orphaned_chunks(self, source_url: str) -> list[str]:
        """
        Identify orphaned chunks that no longer have corresponding content.

        For now, this is a simple implementation that could be enhanced with
        more sophisticated logic to detect truly orphaned chunks.

        Args:
            source_url: Source URL to check for orphaned chunks

        Returns:
            List of orphaned chunk IDs
        """
        if not self.vector_service:
            return []

        try:
            # Simple heuristic: chunks are considered orphaned if they're very old
            # In a more sophisticated implementation, we'd compare against current content
            orphaned_ids: list[str] = []

            # For now, return empty list (no orphaned chunks detected)
            # This can be enhanced later with timestamp-based logic or content comparison

            logger.debug(
                f"Identified {len(orphaned_ids)} orphaned chunks for {source_url}"
            )
            return orphaned_ids

        except Exception as e:
            logger.warning(f"Error identifying orphaned chunks for {source_url}: {e}")
            return []

    async def cleanup_orphaned_chunks(self, chunk_ids: list[str]) -> int:
        """
        Clean up orphaned chunks by deleting them from the vector database.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks successfully deleted
        """
        if not self.vector_service or not chunk_ids:
            return 0

        try:
            # Use the vector service to delete chunks by IDs
            deleted_count = await self.vector_service.operations.delete_chunks_by_ids(
                chunk_ids
            )

            logger.info(f"Successfully deleted {deleted_count} orphaned chunks")
            return int(deleted_count)

        except Exception as e:
            logger.error(f"Error cleaning up orphaned chunks: {e}")
            return 0
