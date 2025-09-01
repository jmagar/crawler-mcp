"""
Data models for web crawling operations.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Reusable validator function
def calculate_word_count_validator(v: int, info: Any) -> int:
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
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("word_count", mode="before")
    @classmethod
    def validate_word_count(cls, v: int, info: Any) -> int:
        """Calculate word count from content if not provided."""
        if v == 0 and info.data and "content" in info.data:
            content = info.data["content"]
            if content and isinstance(content, str):
                return len(content.split())
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
        if isinstance(v, str):
            return [v]
        return v


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
    start_time: datetime = Field(default_factory=datetime.utcnow)
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
