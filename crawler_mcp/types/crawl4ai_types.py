"""Type definitions for crawl4ai integration.

This module provides type-safe interfaces for crawl4ai components
to improve type checking without relying on external library stubs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import Enum
from typing import (
    Any,
    Literal,
    Protocol,
    overload,
    runtime_checkable,
)


class MarkdownGenerationResult(Protocol):
    """Protocol for crawl4ai MarkdownGenerationResult."""

    raw_markdown: str
    markdown_with_citations: str
    references_markdown: str
    fit_markdown: str | None
    fit_html: str | None


class Crawl4aiCrawlResult(Protocol):
    """Protocol for crawl4ai CrawlResult."""

    url: str
    success: bool
    status_code: int | None
    error_message: str | None
    markdown: str | MarkdownGenerationResult
    html: str
    cleaned_html: str
    media: dict[str, Any]
    links: dict[str, Any]
    metadata: dict[str, Any]


class AsyncWebCrawler(Protocol):
    """Protocol for crawl4ai AsyncWebCrawler."""

    async def __aenter__(self) -> AsyncWebCrawler: ...

    async def __aexit__(self, *args: Any) -> None: ...

    @overload
    async def arun(
        self,
        url: str,
        config: Any | None = None,
        *,
        stream: Literal[False] | None = None,
        **kwargs: Any,
    ) -> Crawl4aiCrawlResult: ...

    @overload
    async def arun(
        self,
        url: str,
        config: Any | None = None,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[Crawl4aiCrawlResult]: ...

    async def arun(
        self, url: str, config: Any | None = None, **kwargs: Any
    ) -> Crawl4aiCrawlResult | AsyncIterator[Crawl4aiCrawlResult]: ...


class BrowserConfig(Protocol):
    """Protocol for crawl4ai BrowserConfig."""

    def __init__(
        self, headless: bool = True, verbose: bool = False, **kwargs: Any
    ) -> None: ...


# Strategy protocols for content filtering and markdown generation
@runtime_checkable
class ContentFilterStrategy(Protocol):
    """Protocol for crawl4ai content filter strategies."""

    def filter_content(self, content: str, **kwargs: Any) -> str: ...


@runtime_checkable
class PruningContentFilter(ContentFilterStrategy, Protocol):
    """Protocol for crawl4ai PruningContentFilter."""

    def __init__(
        self,
        threshold: float = 0.48,
        threshold_type: str = "fixed",
        min_word_threshold: int = 0,
        **kwargs: Any,
    ) -> None: ...


@runtime_checkable
class MarkdownGenerationStrategy(Protocol):
    """Protocol for crawl4ai markdown generation strategies."""

    def generate_markdown(
        self, html: str, base_url: str = "", **kwargs: Any
    ) -> MarkdownGenerationResult: ...


@runtime_checkable
class DefaultMarkdownGenerator(MarkdownGenerationStrategy, Protocol):
    """Protocol for crawl4ai DefaultMarkdownGenerator."""

    def __init__(
        self, content_filter: ContentFilterStrategy | None = None, **kwargs: Any
    ) -> None: ...


# Extraction strategy protocols
@runtime_checkable
class ExtractionStrategy(Protocol):
    """Base protocol for crawl4ai extraction strategies."""

    def extract(self, html: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class CosineStrategy(ExtractionStrategy, Protocol):
    """Protocol for crawl4ai CosineStrategy."""

    def __init__(
        self,
        semantic_filter: str | None = None,
        word_count_threshold: int = 10,
        max_dist: float = 0.2,
        linkage_method: str = "ward",
        top_k: int = 3,
        model_name: str = "BAAI/bge-small-en-v1.5",
        **kwargs: Any,
    ) -> None: ...


@runtime_checkable
class LLMExtractionStrategy(ExtractionStrategy, Protocol):
    """Protocol for crawl4ai LLMExtractionStrategy."""

    def __init__(
        self,
        provider: str | None = None,
        api_token: str | None = None,
        instruction: str | None = None,
        schema: dict[str, Any] | None = None,
        extraction_type: str = "schema",
        apply_chunking: bool = True,
        **kwargs: Any,
    ) -> None: ...


# Deep crawling protocols
class CacheMode(Enum):
    """Cache mode enumeration for crawl4ai."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    BYPASS = "bypass"


@runtime_checkable
class URLPatternFilter(Protocol):
    """Protocol for crawl4ai URL pattern filters."""

    def __init__(
        self,
        patterns: list[str],
        action: str = "include",  # "include" or "exclude"
        **kwargs: Any,
    ) -> None: ...

    def should_include(self, url: str) -> bool: ...


@runtime_checkable
class KeywordRelevanceScorer(Protocol):
    """Protocol for crawl4ai keyword relevance scorer."""

    def __init__(
        self, keywords: list[str], threshold: float = 0.6, **kwargs: Any
    ) -> None: ...

    def score(self, content: str) -> float: ...


@runtime_checkable
class FilterChain(Protocol):
    """Protocol for crawl4ai filter chain."""

    def __init__(self, filters: list[Any]) -> None: ...

    def apply(self, item: Any) -> bool: ...


@runtime_checkable
class DeepCrawlStrategy(Protocol):
    """Protocol for crawl4ai deep crawl strategy."""

    def __init__(
        self,
        max_depth: int = 3,
        filter_chain: FilterChain | None = None,
        scorer: Any | None = None,
        **kwargs: Any,
    ) -> None: ...


# Updated CrawlerRunConfig with better typing
class CrawlerRunConfigAdvanced(Protocol):
    """Protocol for crawl4ai CrawlerRunConfig."""

    def __init__(
        self,
        deep_crawl_strategy: DeepCrawlStrategy | None = None,
        stream: bool = False,
        cache_mode: CacheMode | str | None = None,
        page_timeout: float | None = None,
        markdown_generator: MarkdownGenerationStrategy | None = None,
        content_filter: ContentFilterStrategy | None = None,
        extraction_strategy: ExtractionStrategy | None = None,
        **kwargs: Any,
    ) -> None: ...


# Runtime import helpers for type-safe access to crawl4ai
# These functions provide type-safe wrappers around dynamic imports

# Initialize variables that will be set based on import success
DefaultMarkdownGeneratorImpl: type[DefaultMarkdownGenerator] | None
PruningContentFilterImpl: type[PruningContentFilter] | None
MarkdownGenerationResultImpl: type[MarkdownGenerationResult] | None

# For actual runtime usage, we just cast the imports to our protocols
# This maintains type safety while allowing runtime flexibility
try:
    from crawl4ai import (
        DefaultMarkdownGenerator as _DefaultMarkdownGenerator,  # type: ignore
    )
    from crawl4ai import (
        PruningContentFilter as _PruningContentFilter,  # type: ignore
    )
    from crawl4ai.models import (
        MarkdownGenerationResult as _MarkdownGenerationResult,  # type: ignore
    )

    # Type-safe aliases that satisfy our protocols
    DefaultMarkdownGeneratorImpl = _DefaultMarkdownGenerator
    PruningContentFilterImpl = _PruningContentFilter
    MarkdownGenerationResultImpl = _MarkdownGenerationResult

except ImportError:
    # Fallback for when crawl4ai is not available
    DefaultMarkdownGeneratorImpl = None  # type: ignore[assignment]
    PruningContentFilterImpl = None  # type: ignore[assignment]
    MarkdownGenerationResultImpl = None  # type: ignore[assignment]


# Type aliases for cleaner imports
Crawl4aiResult = Crawl4aiCrawlResult
Crawl4aiMarkdownResult = MarkdownGenerationResult

# Export the implementation classes for runtime use
__all__ = [
    "AsyncWebCrawler",
    "BrowserConfig",
    "CacheMode",
    "ContentFilterStrategy",
    "CosineStrategy",
    "Crawl4aiCrawlResult",
    "Crawl4aiMarkdownResult",
    "Crawl4aiResult",
    "CrawlerRunConfigAdvanced",
    "DeepCrawlStrategy",
    "DefaultMarkdownGenerator",
    "DefaultMarkdownGeneratorImpl",
    "ExtractionStrategy",
    "FilterChain",
    "KeywordRelevanceScorer",
    "LLMExtractionStrategy",
    "MarkdownGenerationResult",
    "MarkdownGenerationResultImpl",
    "MarkdownGenerationStrategy",
    "PruningContentFilter",
    "PruningContentFilterImpl",
    "URLPatternFilter",
]
