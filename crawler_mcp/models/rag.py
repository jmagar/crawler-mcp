"""
Data models for RAG (Retrieval-Augmented Generation) operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .crawl import calculate_word_count_validator


class DocumentChunk(BaseModel):
    model_config = ConfigDict()

    """A chunk of document content with metadata."""

    id: str
    content: str
    embedding: list[float] | None = None
    source_url: str
    source_title: str | None = None
    chunk_index: int = 0
    word_count: int = 0
    char_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Deduplication fields
    content_hash: str | None = None  # SHA256 hash of content for change detection
    previous_hash: str | None = None  # Previous content hash if content changed
    first_seen: datetime = Field(default_factory=datetime.utcnow)  # When first crawled
    last_modified: datetime = Field(
        default_factory=datetime.utcnow
    )  # When content last changed

    # Crawl provenance tracking
    seed_url: str | None = None  # Original URL that initiated the crawl
    crawl_session_id: str | None = (
        None  # Unique ID linking all documents from same crawl
    )
    discovery_method: str | None = (
        None  # How this URL was found: "seed", "sitemap", "link_following", "manual"
    )

    _validate_word_count = field_validator("word_count", mode="before")(
        calculate_word_count_validator
    )

    @field_validator("char_count", mode="before")
    @classmethod
    def calculate_char_count(cls, v: int, info: Any) -> int:
        if v == 0 and info.data and "content" in info.data:
            content = info.data["content"]
            if content:
                return len(content)
        return v


class SearchMatch(BaseModel):
    model_config = ConfigDict()

    """A search result match with similarity score."""

    document: DocumentChunk
    score: float = Field(ge=0.0, le=1.0)
    relevance: str = Field(default="medium")  # low, medium, high
    highlighted_content: str | None = None

    @field_validator("relevance", mode="before")
    @classmethod
    def calculate_relevance(cls, v: str, info: Any) -> str:
        if info.data and "score" in info.data:
            score = info.data["score"]
            if score >= 0.8:
                return "high"
            elif score >= 0.6:
                return "medium"
            else:
                return "low"
        return v


class RagQuery(BaseModel):
    model_config = ConfigDict()

    """Query for RAG search operations."""

    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_filters: list[str] | None = None
    date_range: tuple[datetime, datetime] | None = None
    include_content: bool = True
    include_metadata: bool = True
    rerank: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class EmbeddingResult(BaseModel):
    model_config = ConfigDict()

    """Result of embedding generation."""

    text: str
    embedding: list[float]
    model: str
    dimensions: int
    processing_time: float = 0.0

    @field_validator("dimensions", mode="before")
    @classmethod
    def calculate_dimensions(cls, v: int, info: Any) -> int:
        if v == 0 and info.data and "embedding" in info.data:
            embedding = info.data["embedding"]
            if embedding:
                return len(embedding)
        return v


class RagResult(BaseModel):
    model_config = ConfigDict()

    """Result of a RAG query operation."""

    query: str
    matches: list[SearchMatch] = Field(default_factory=list)
    total_matches: int = 0
    processing_time: float = 0.0
    embedding_time: float = 0.0
    search_time: float = 0.0
    rerank_time: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def has_high_confidence_matches(self) -> bool:
        """Check if any matches have high confidence."""
        return any(match.relevance == "high" for match in self.matches)

    @property
    def average_score(self) -> float:
        """Calculate average similarity score."""
        if not self.matches:
            return 0.0
        return sum(match.score for match in self.matches) / len(self.matches)

    @property
    def best_match_score(self) -> float:
        """Get the highest similarity score."""
        if not self.matches:
            return 0.0
        return max(match.score for match in self.matches)
