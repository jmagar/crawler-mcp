"""
Data models for web crawling operations.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


def get_embedding_dim() -> int:
    """Get embedding dimension from configuration at runtime."""
    from crawler_mcp.settings import get_settings

    settings = get_settings()
    return settings.embedding_dimension


# Reusable validator function
def calculate_word_count_validator(v: int, info: ValidationInfo) -> int:
    """Pydantic validator to calculate word count from content."""
    if v == 0 and info.data and "content" in info.data:
        content = info.data["content"]
        if content:
            return len(content.split())
    return v


class CrawlStatus(str, Enum):
    """Status of a crawl operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PageContent(BaseModel):
    """Content extracted from a single page."""

    model_config = ConfigDict()

    url: str
    title: str | None = None
    content: str
    markdown: str | None = None
    html: str | None = None
    links: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    word_count: int = 0
    links_count: int = Field(default=0, ge=0)
    images_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    embedding: list[float] | None = Field(default=None, repr=False)

    _validate_word_count = field_validator("word_count", mode="before")(
        calculate_word_count_validator
    )

    @field_validator("links_count", mode="before")
    @classmethod
    def derive_links_count(cls, v: int | None, info: ValidationInfo) -> int:
        """Derive links_count from links list if not explicitly set, with consistency validation."""
        if info.data and "links" in info.data:
            links = info.data.get("links", [])
            if not isinstance(links, list):
                links = []
            actual_count = len(links)

            if v is None or v == 0:
                return actual_count

            if v != actual_count:
                logging.getLogger(__name__).debug(
                    f"Links count mismatch: provided={v}, actual={actual_count}"
                )

        return v if v is not None else 0

    @field_validator("images_count", mode="before")
    @classmethod
    def derive_images_count(cls, v: int | None, info: ValidationInfo) -> int:
        """Derive images_count from images list if not explicitly set."""
        if info.data and "images" in info.data:
            images = info.data.get("images", [])
            if not isinstance(images, list):
                images = []

            if v is None or v == 0:
                return len(images)

        return v if v is not None else 0

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimension(
        cls, v: list[float] | None, info: ValidationInfo
    ) -> list[float] | None:
        """Validate embedding has correct dimension and valid values with hardened security."""
        if v is not None:
            # Basic type and structure validation
            if not isinstance(v, list):
                raise ValueError(f"Embedding must be a list, got {type(v)}")

            if len(v) == 0:
                raise ValueError("Embedding cannot be empty")

            # Hardened dimension bounds validation
            if len(v) > 4096:
                raise ValueError(
                    f"Embedding dimension {len(v)} exceeds maximum allowed (4096). "
                    "This may indicate corrupted data or memory exhaustion attack."
                )
            if len(v) < 32:
                raise ValueError(
                    f"Embedding dimension {len(v)} is suspiciously small (minimum 32). "
                    "Check if embedding model is configured correctly."
                )

            # Check for finite values and valid float types with value bounds
            extreme_values = []
            for i, val in enumerate(v):
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Embedding value at index {i} must be numeric, got {type(val)}: {val}"
                    )
                if not math.isfinite(val):
                    raise ValueError(
                        f"Embedding contains non-finite value at index {i}: {val}"
                    )
                # Check for reasonable value bounds (most embeddings are in [-10, 10] range)
                if abs(val) > 100.0:
                    extreme_values.append((i, val))

            if extreme_values:
                raise ValueError(
                    f"Embedding contains {len(extreme_values)} extreme values (>100): "
                    f"{extreme_values[:3]}{'...' if len(extreme_values) > 3 else ''}. "
                    "This may indicate corrupted or unnormalized embeddings."
                )

            # Detect suspicious patterns
            v_set = set(v)
            if len(v_set) == 1:
                raise ValueError(
                    f"Embedding has all identical values ({next(iter(v_set))}). "
                    "This indicates degenerate or corrupted embedding."
                )
            if len(v_set) < len(v) * 0.1:  # Less than 10% unique values
                raise ValueError(
                    f"Embedding has suspiciously few unique values ({len(v_set)}/{len(v)}). "
                    "This may indicate corrupted or quantized embedding."
                )

            # Validate vector norm (embeddings should have reasonable magnitude)
            norm = math.sqrt(sum(x * x for x in v))
            if norm == 0.0:
                raise ValueError("Embedding has zero norm (all values are zero)")
            if norm < 1e-6:
                raise ValueError(
                    f"Embedding has suspiciously small norm ({norm:.2e}). "
                    "This may indicate corrupted or scaled embedding."
                )
            if norm > 1000.0:
                raise ValueError(
                    f"Embedding has suspiciously large norm ({norm:.2f}). "
                    "This may indicate unnormalized or corrupted embedding."
                )

            # Get expected dimension from runtime accessor with proper error handling
            try:
                expected_dim = get_embedding_dim()
            except Exception as e:
                raise ValueError(
                    f"Failed to get expected embedding dimension from configuration: {e}. "
                    "Ensure EMBEDDING_DIMENSION environment variable is properly set."
                ) from e

            # Validate dimension bounds
            if expected_dim <= 0:
                raise ValueError(
                    f"Invalid expected embedding dimension: {expected_dim}. "
                    "EMBEDDING_DIMENSION must be a positive integer."
                )

            if len(v) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dim}, got {len(v)}. "
                    f"To fix: Either recompute embeddings with dimension {expected_dim} or "
                    f"update EMBEDDING_DIMENSION environment variable to {len(v)}."
                )
        return v


class CrawlRequest(BaseModel):
    """Request for crawling operation."""

    model_config = ConfigDict()

    url: str | list[str]
    max_pages: int | None = Field(default=100, ge=1, le=1000)
    max_depth: int | None = Field(default=3, ge=1, le=10)
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    extraction_strategy: str | None = Field(default=None)
    wait_for: str | None = None
    remove_overlay_elements: bool = True
    extract_media: bool = False
    include_raw_html: bool = False
    session_id: str | None = None
    chunking_strategy: str | None = None
    chunking_options: dict[str, Any] | None = None

    # Content Filtering Options
    excluded_tags: list[str] | None = Field(
        default=None, description="HTML tags to exclude from content extraction"
    )
    excluded_selectors: list[str] | None = Field(
        default=None, description="CSS selectors to exclude from content extraction"
    )
    content_selector: str | None = Field(
        default=None, description="CSS selector to focus on main content area"
    )
    pruning_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Content relevance threshold for filtering",
    )
    min_word_threshold: int | None = Field(
        default=None,
        ge=5,
        le=100,
        description="Minimum words required for content blocks",
    )
    prefer_fit_markdown: bool | None = Field(
        default=None,
        description="Prefer filtered fit_markdown over raw_markdown (None = use global setting)",
    )

    @field_validator("url")
    @classmethod
    def validate_urls(cls, v: str | list[str]) -> list[str]:
        urls = [v] if isinstance(v, str) else v
        if not urls:
            raise ValueError("url must be non-empty")
        return urls


class CrawlStatistics(BaseModel):
    model_config = ConfigDict()

    """Statistics for a crawl operation."""
    total_pages_requested: int = 0
    total_pages_crawled: int = 0
    total_pages_failed: int = 0
    total_bytes_downloaded: int = 0
    average_page_size: float = 0.0
    crawl_duration_seconds: float = 0.0
    pages_per_second: float = 0.0
    unique_domains: int = 0
    total_links_discovered: int = 0
    total_images_found: int = 0
    error_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def attempted_pages(self) -> int:
        """Total number of pages attempted (crawled + failed)."""
        return self.total_pages_crawled + self.total_pages_failed


class CrawlResult(BaseModel):
    model_config = ConfigDict()

    """Result of a crawl operation."""
    request_id: str
    status: CrawlStatus
    urls: list[str]
    pages: list[PageContent] = Field(default_factory=list)
    statistics: CrawlStatistics = Field(default_factory=CrawlStatistics)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage of actual pages attempted."""
        if self.statistics.attempted_pages == 0:
            return 0.0
        return (
            self.statistics.total_pages_crawled / self.statistics.attempted_pages
        ) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if crawl is complete."""
        return self.status in [
            CrawlStatus.COMPLETED,
            CrawlStatus.FAILED,
            CrawlStatus.CANCELLED,
        ]
