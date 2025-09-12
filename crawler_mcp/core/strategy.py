"""
Main orchestrator for optimized high-performance web crawler.

This module implements the main crawler orchestrator that uses Crawl4AI's
deep crawling APIs to discover and crawl multiple pages efficiently.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

try:  # Prefer BestFirst when available
    from crawl4ai.deep_crawling import BestFirstCrawlingStrategy as _C4BestFirst
except Exception:
    _C4BestFirst = None
from crawl4ai.deep_crawling.filters import FilterChain

try:
    from crawl4ai.deep_crawling.filters import (
        ContentTypeFilter as _C4ContentTypeFilter,  # type: ignore
    )
except Exception:
    _C4ContentTypeFilter = None
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from crawler_mcp.clients.qdrant_http_client import QdrantClient
from crawler_mcp.clients.tei_client import TEIEmbeddingsClient

# from crawler_mcp.core.vectors.collections import CollectionManager
from crawler_mcp.core.batch_utils import pack_texts_into_batches
from crawler_mcp.factories.browser_factory import BrowserFactory
from crawler_mcp.factories.content_extractor import ContentExtractorFactory
from crawler_mcp.models.crawl import PageContent
from crawler_mcp.models.responses import OptimizedCrawlResponse
from crawler_mcp.processing.result_converter import ResultConverter
from crawler_mcp.settings import CrawlerSettings, get_settings
from crawler_mcp.utils.monitoring import PerformanceMonitor

# Prefer official Crawl4AI filters/scorers when available
try:  # DomainFilter may not exist in older versions
    from crawl4ai.deep_crawling.filters import (
        DomainFilter as _C4DomainFilter,  # type: ignore
    )
except Exception:
    _C4DomainFilter = None

try:
    from crawl4ai.deep_crawling.scorers import (  # type: ignore
        KeywordRelevanceScorer as _C4KeywordRelevanceScorer,
    )
except Exception:
    _C4KeywordRelevanceScorer = None


class ExclusionFilter:
    """Filter to exclude URLs matching specific patterns."""

    def __init__(self, patterns: list[str]):
        """
        Initialize exclusion filter with regex patterns.

        Args:
            patterns: List of regex patterns to exclude
        """
        import re

        self.patterns = [re.compile(p) for p in patterns if p]

    def should_include(self, url: str) -> bool:
        """
        Return True to include, False to exclude.
        """
        return all(not pattern.search(url) for pattern in self.patterns)

    # Back-compat alias if anything already calls .apply()
    def apply(self, url: str) -> bool:  # pragma: no cover
        return self.should_include(url)


class DomainIncludeFilter:
    """Filter to include only URLs from the specified domain."""

    def __init__(self, domain: str):
        self.domain = domain.lower()

    def should_include(self, url: str) -> bool:
        try:
            if not url:
                return False
            # Treat root-relative paths as internal
            if url.startswith("/"):
                return True
            parsed = urlparse(url)
            # If no scheme/netloc, allow (likely relative link)
            if not parsed.scheme and not parsed.netloc:
                return True
            return parsed.netloc.lower() == self.domain
        except Exception:
            return False

    def apply(self, url: str) -> bool:  # pragma: no cover
        return self.should_include(url)


class KeywordURLScorer:
    """Simple keyword-based URL relevance scorer.

    Scores based on presence of any configured keywords in the URL/path.
    Returns a float in [0, 1].
    """

    def __init__(self, keywords: list[str], weight: float = 1.0):
        self.keywords = [k.lower() for k in keywords if k]
        self.weight = max(0.0, float(weight))

    def score(self, content: str) -> float:  # content = URL string here
        try:
            url_l = (content or "").lower()
            if not url_l or not self.keywords:
                return 0.0
            hits = sum(1 for k in self.keywords if k in url_l)
            base = min(1.0, hits / max(1, len(self.keywords)))
            return max(0.0, min(1.0, base * self.weight))
        except Exception:
            return 0.0


class CrawlOrchestrator:
    """
    High-performance web crawler orchestrator using Crawl4AI's deep crawling APIs.

    This orchestrator uses Crawl4AI's documented BFSDeepCrawlStrategy to handle
    both URL discovery and parallel crawling in a single unified process.
    """

    # Class-level cache mode mapping to avoid recreation on every method call
    _CACHE_MODES: ClassVar[dict[str, CacheMode]] = {
        "enabled": CacheMode.ENABLED,
        "bypass": CacheMode.BYPASS,
        "disabled": CacheMode.DISABLED,
        "adaptive": CacheMode.ENABLED,  # Default to enabled for adaptive
        "read_only": CacheMode.READ_ONLY,
        "write_only": CacheMode.WRITE_ONLY,
    }

    def __init__(
        self, settings: CrawlerSettings, overrides: dict[str, Any] | None = None
    ):
        """
        Initialize the crawler orchestrator.

        Args:
            settings: Global settings instance
            overrides: Optional runtime configuration overrides
        """
        self.settings = settings
        self.config = settings  # Alias for backward compatibility
        self.overrides = overrides or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.browser_factory = BrowserFactory(settings, overrides)
        self.content_extractor = ContentExtractorFactory(settings, overrides)
        self.result_converter = ResultConverter(settings, overrides)
        self.monitor = PerformanceMonitor()

        # State management
        self._last_pages: list[PageContent] = []
        self._hooks: dict[str, Any] = {}

        # Stats
        self.stats = {
            "pages_crawled": 0,
            "pages_failed": 0,
            "start_time": None,
            "end_time": None,
        }

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value from overrides or settings."""
        return self.overrides.get(key, getattr(self.settings, key, default))

    def _serialize_enum_value(self, value: Any) -> str:
        """Convert enum values to their serializable string representation."""
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)

    async def start(self) -> None:
        """Initialize the orchestrator."""
        self.logger.info("CrawlOrchestrator initialized")

    async def close(self) -> None:
        """Clean up resources."""
        self.logger.info("CrawlOrchestrator shut down")

    def set_hook(self, event: str, callback: Any) -> None:
        """Set a monitoring hook for events."""
        self._hooks[event] = callback

    def _trigger_hook(self, event: str, **kwargs: Any) -> None:
        """Trigger a monitoring hook."""
        if event in self._hooks:
            try:
                self._hooks[event](**kwargs)
            except Exception as e:
                self.logger.debug("Hook %s failed: %s", event, e)

    async def crawl(
        self,
        start_url: str,
        max_urls: int | None = None,
        **kwargs: Any,
    ) -> OptimizedCrawlResponse:
        """
        Crawl a website using Crawl4AI's deep crawling strategy.

        Args:
            start_url: Starting URL to crawl
            max_urls: Maximum number of pages to crawl
            **kwargs: Additional crawling parameters

        Returns:
            OptimizedCrawlResponse with crawl results
        """
        self.stats["start_time"] = time.time()
        # Apply page limit: prefer explicit argument, else global MAX_PAGES
        max_urls = max_urls or self.settings.max_pages

        # Get streaming setting from instance settings first, fallback to global
        stream = getattr(self.settings, "enable_streaming", None)
        if stream is None:
            stream = get_settings().enable_streaming

        self.logger.info(f"Starting deep crawl of {start_url} (max: {max_urls} pages)")

        # Create domain-based URL filter
        domain = urlparse(start_url).netloc
        self.logger.info(f"Creating URL filter for domain: {domain}")
        # Include only this domain (support absolute and relative)
        if _C4DomainFilter is not None:
            domain_include_filter = _C4DomainFilter(allowed_domains=[domain])
        else:
            domain_include_filter = DomainIncludeFilter(domain)

        # Proactive exclusion for low-value docs URLs (pre-crawl) via regex
        builtin_exclude_patterns = [r".*__.*__.*", r".*__init__.*"]
        builtin_exclude_filter = ExclusionFilter(builtin_exclude_patterns)

        # Create exclusion filter using config patterns
        exclusion_patterns = self.get_config_value("crawl_exclude_url_patterns", [])
        # Keep regex-based exclusion for user patterns
        exclusion_filter = ExclusionFilter(exclusion_patterns)
        self.logger.info(
            f"Created exclusion filter with {len(exclusion_patterns)} patterns"
        )

        # Optional URL relevance scorer (pre-crawl)
        scorer = None
        if self.get_config_value("enable_url_scoring", False):
            try:
                keywords = self.get_config_value("url_score_keywords", [])
                weight = 1.0
                if _C4KeywordRelevanceScorer is not None:
                    scorer = _C4KeywordRelevanceScorer(keywords=keywords, weight=weight)
                else:
                    scorer = KeywordURLScorer(keywords=keywords, weight=weight)
                self.logger.info(
                    f"URL scoring enabled: {len(keywords)} keywords, threshold={self.get_config_value('url_score_threshold', 0.6)}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize URL scorer: {e}")
                scorer = None

        # Create deep crawl strategy with both inclusion and exclusion filters
        # Choose BestFirst when scoring is enabled and implementation is available
        # Build filter chain with optional content-type filter
        filters_list = [domain_include_filter, builtin_exclude_filter, exclusion_filter]
        allowed_ctypes = (
            self.get_config_value("allowed_content_types", ["text/html"]) or []
        )
        if _C4ContentTypeFilter is not None and allowed_ctypes:
            try:
                filters_list.append(_C4ContentTypeFilter(allowed_types=allowed_ctypes))
                self.logger.info(
                    f"Added ContentTypeFilter to filter chain: {allowed_ctypes}"
                )
            except Exception as e:
                self.logger.debug(f"Failed to add ContentTypeFilter: {e}")
        filter_chain = FilterChain(filters_list)

        if scorer and _C4BestFirst is not None:
            self.logger.info(
                f"Creating BestFirst strategy: max_depth=3, max_pages={max_urls} (scoring enabled)"
            )
            deep_crawl_strategy = _C4BestFirst(
                max_depth=3,
                include_external=False,
                max_pages=max_urls,
                filter_chain=filter_chain,
                url_scorer=scorer,
            )
        else:
            # Only apply a non-zero threshold when a scorer is active
            score_thr = (
                self.get_config_value("url_score_threshold", 0.0) if scorer else 0.0
            )
            self.logger.info(
                f"Creating BFS strategy: max_depth=3, max_pages={max_urls}, threshold={score_thr} "
                + ("(scoring enabled)" if scorer else "(scoring disabled)")
            )
            deep_crawl_strategy = BFSDeepCrawlStrategy(
                max_depth=3,
                include_external=False,
                max_pages=max_urls,
                filter_chain=filter_chain,
                url_scorer=scorer,
                score_threshold=score_thr,
            )

        # Configure crawler with performance optimizations
        browser_config = self._get_optimized_browser_config(start_url)
        config = self._create_optimized_crawler_config(
            deep_crawl_strategy=deep_crawl_strategy, url=start_url, stream=stream
        )

        # Debug configuration that might cause result.success=False
        self.logger.info(
            f"Browser config - ignore_https_errors: {getattr(browser_config, 'ignore_https_errors', 'N/A')}"
        )
        self.logger.info(
            f"Crawler config - page_timeout: {getattr(config, 'page_timeout', 'N/A')}ms"
        )
        self.logger.info(
            f"Crawler config - word_count_threshold: {getattr(config, 'word_count_threshold', 'N/A')}"
        )
        self.logger.info(
            f"Crawler config - cache_mode: {getattr(config, 'cache_mode', 'N/A')}"
        )
        self.logger.info(
            f"Follow redirects: {self.get_config_value('follow_redirects', True)}"
        )
        self.logger.info(f"Max redirects: {self.get_config_value('max_redirects', 5)}")

        # Execute crawl with enhanced logging
        try:
            self.logger.info(
                f"Executing deep crawl with {type(deep_crawl_strategy).__name__}"
            )
            self.logger.info(
                f"Memory threshold: {self.get_config_value('memory_threshold_percent', 80)}%, Max sessions: {self.get_config_value('max_session_permit', 5)}"
            )
            self.logger.info(
                f"Semaphore count: {self.get_config_value('crawl_semaphore_count', 3)}, Delays: {self.get_config_value('mean_request_delay', 1.0)}s-{self.get_config_value('max_request_delay_range', 2.0)}s"
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                if stream:
                    # Stream mode - process results as they arrive with progress logging
                    crawl_results = []
                    page_count = 0
                    start_time = time.time()

                    self.logger.info("Starting streaming crawl...")
                    try:
                        # Stream mode - process results as they arrive
                        crawl_generator = await crawler.arun(start_url, config=config)
                        self.logger.info(
                            f"Got crawl generator: {type(crawl_generator)}"
                        )

                        async for result in crawl_generator:
                            try:
                                # Validate result is actually a CrawlResult object
                                if result is None:
                                    self.logger.warning(
                                        "Received None result from stream"
                                    )
                                    continue

                                # Check for required attributes
                                if not hasattr(result, "url") or not hasattr(
                                    result, "success"
                                ):
                                    self.logger.warning(
                                        f"Invalid result object type: {type(result)}, "
                                        f"has url: {hasattr(result, 'url')}, "
                                        f"has success: {hasattr(result, 'success')}"
                                    )
                                    continue

                                page_count += 1
                            except AttributeError as e:
                                self.logger.error(f"Result processing error: {e}")
                                continue

                            # Enhanced per-page logging
                            elapsed = time.time() - start_time
                            rate = page_count / elapsed if elapsed > 0 else 0

                            if result.success:
                                # Use fit_markdown if available (filtered content), fallback to raw_markdown
                                markdown_content = ""
                                if hasattr(result, "markdown") and result.markdown:
                                    if (
                                        hasattr(result.markdown, "fit_markdown")
                                        and result.markdown.fit_markdown
                                    ):
                                        markdown_content = result.markdown.fit_markdown
                                    elif hasattr(result.markdown, "raw_markdown"):
                                        markdown_content = (
                                            result.markdown.raw_markdown or ""
                                        )

                                content_length = len(markdown_content)
                                status = getattr(result, "status_code", None)
                                redirected_url = getattr(result, "redirected_url", None)
                                original_url = result.url

                                # Enhanced redirect logging
                                redirect_info = ""
                                if redirected_url and redirected_url != original_url:
                                    redirect_info = f" (redirected from {original_url})"

                                should_append = True
                                if status and not (200 <= int(status) < 300):
                                    self.logger.info(
                                        f"⚠️  Skipping (HTTP {status}) Page {page_count}: {redirected_url or result.url}{redirect_info} ({content_length} chars, {rate:.1f} pages/sec)"
                                    )
                                    should_append = False
                                elif (
                                    content_length <= 50
                                ):  # Increased threshold from 5 to 50
                                    self.logger.info(
                                        f"⚠️  Low content warning - Page {page_count}: {redirected_url or result.url}{redirect_info} ({content_length} chars, {rate:.1f} pages/sec)"
                                    )
                                    should_append = False
                                else:
                                    self.logger.info(
                                        f"✅ Page {page_count}: {redirected_url or result.url}{redirect_info} ({content_length} chars, {rate:.1f} pages/sec)"
                                    )
                                if should_append:
                                    crawl_results.append(result)
                                self._trigger_hook(
                                    "page_crawled",
                                    url=result.url,
                                    content_length=content_length,
                                    crawl_time=elapsed,
                                )
                            else:
                                # Safely get attributes with defaults
                                url = getattr(result, "url", "unknown")
                                error_msg = getattr(
                                    result, "error_message", "Unknown error"
                                )
                                status = getattr(result, "status_code", "unknown")
                                redirected_url = getattr(result, "redirected_url", None)
                                html_length = len(getattr(result, "html", "") or "")

                                # Enhanced failure logging
                                redirect_info = ""
                                if redirected_url and redirected_url != url:
                                    redirect_info = f" (redirected to {redirected_url})"

                                self.logger.warning(
                                    f"❌ Page {page_count}: {url}{redirect_info} - Status: {status}, HTML length: {html_length}, Error: {error_msg}"
                                )

                                # Critical: This failed result won't contribute to link discovery
                                if page_count == 1:
                                    self.logger.error(
                                        "🚨 CRITICAL: First page failed - this will prevent link discovery and deep crawling!"
                                    )
                                self._trigger_hook(
                                    "page_failed",
                                    url=url,
                                    error=error_msg,
                                    error_type="crawl_failed",
                                )

                            # Update stats incrementally for O(1) performance
                            if result.success:
                                self.stats["pages_crawled"] += 1
                            else:
                                self.stats["pages_failed"] += 1

                            # Progress logging every 10 pages
                            if page_count % 10 == 0:
                                self.logger.info(
                                    f"Progress: {page_count} pages processed, {rate:.1f} pages/sec"
                                )

                    except (GeneratorExit, asyncio.CancelledError):
                        self.logger.info("Stream iteration was cancelled/interrupted")
                        # This is expected when the generator is cleaned up
                        pass
                    except Exception:
                        self.logger.exception("Error during stream iteration")
                        raise
                    finally:
                        self.logger.info(
                            f"Stream iteration ended after {page_count} pages"
                        )

                else:
                    # Non-stream mode - get all results at once
                    self.logger.info("Starting non-streaming crawl...")
                    crawl_results = await crawler.arun(start_url, config=config)
                    if not isinstance(crawl_results, list):
                        crawl_results = [crawl_results]

                    def _content_len(r: Any) -> int:
                        try:
                            md = getattr(r, "markdown", None)
                            if isinstance(md, str):
                                return len(md)
                            if md and getattr(md, "fit_markdown", None):
                                return len(md.fit_markdown)
                            if md and hasattr(md, "raw_markdown"):
                                return len(md.raw_markdown or "")
                        except Exception:
                            return 0
                        return 0

                    filtered_results: list[Any] = []
                    for r in crawl_results:
                        status = getattr(r, "status_code", None)
                        clen = _content_len(r)
                        if status and not (200 <= int(status) < 300):
                            continue
                        if clen <= 50:  # Match streaming mode threshold
                            continue
                        filtered_results.append(r)

                    self.stats["pages_crawled"] = len(
                        [r for r in filtered_results if r.success]
                    )
                    self.stats["pages_failed"] = len(
                        [r for r in crawl_results if not r.success]
                    )

                    # Log results summary with timing
                    current_duration = time.time() - self.stats["start_time"]
                    self.logger.info(
                        f"Crawl completed: {self.stats['pages_crawled']} successful, "
                        f"{self.stats['pages_failed']} failed in {current_duration:.2f}s"
                    )

            self.logger.info(f"Deep crawl completed: {len(crawl_results)} pages")

            # Convert results to our page format
            pages = []
            source_results = (
                filtered_results if "filtered_results" in locals() else crawl_results
            )
            for result in source_results:
                if result.success:
                    page = self.result_converter.crawl4ai_to_page_content(result)
                    pages.append(page)
                else:
                    self._trigger_hook(
                        "page_failed",
                        url=result.url,
                        error=result.error_message,
                        error_type="crawl_failed",
                    )

            self._last_pages = pages

            # Process embeddings if enabled
            if self.get_config_value("enable_embeddings", False) and pages:
                await self._process_embeddings(pages)

            # Process Qdrant upserts if enabled
            if self.get_config_value("enable_qdrant", False) and pages:
                await self._process_qdrant_upserts(pages)

            self.stats["end_time"] = time.time()

            # Calculate crawl duration and log final timing
            duration = self.stats["end_time"] - self.stats["start_time"]
            rate = len(pages) / duration if duration > 0 else 0.0
            self.logger.info(
                f"Crawl completed in {duration:.2f} seconds - "
                f"{len(pages)} pages processed at {rate:.1f} pages/sec"
            )

            # Create metadata with timing info
            metadata = {
                "duration_seconds": duration,
                "pages_per_second": len(pages) / duration if duration > 0 else 0,
                "start_time": self.stats["start_time"],
                "end_time": self.stats["end_time"],
            }

            # Create response
            html_content = "\n\n".join(
                [f"<!-- PAGE: {page.url} -->\n{page.content}" for page in pages]
            )

            return OptimizedCrawlResponse(
                html=html_content,
                status_code=200,
                response_headers={},
                js_execution_result=None,
                screenshot=None,
                pdf_data=None,
                mhtml_data=None,
                get_delayed_content=None,
                downloaded_files=None,
                ssl_certificate=None,
                redirected_url=None,
                network_requests=None,
                console_messages=None,
                pages_crawled=len(pages),
                total_pages=len(crawl_results),
                success=len(pages) > 0,
                metadata=metadata,
            )

        except Exception as e:
            self.stats["end_time"] = time.time()
            duration = self.stats["end_time"] - self.stats.get(
                "start_time", self.stats["end_time"]
            )
            self.logger.error(
                f"Deep crawl failed after {duration:.2f}s: {e}", exc_info=True
            )

            return OptimizedCrawlResponse(
                html="",
                status_code=500,
                response_headers={},
                js_execution_result=None,
                screenshot=None,
                pdf_data=None,
                mhtml_data=None,
                get_delayed_content=None,
                downloaded_files=None,
                ssl_certificate=None,
                redirected_url=None,
                network_requests=None,
                console_messages=None,
                pages_crawled=0,
                total_pages=0,
                success=False,
                error_message=str(e),
                metadata={"duration_seconds": duration, "error": str(e)},
            )

    async def _process_embeddings(self, pages: list[PageContent]) -> None:
        """Process embeddings for crawled pages."""
        if not self.get_config_value("tei_endpoint", self.settings.tei_url):
            self.logger.warning("TEI endpoint not configured, skipping embeddings")
            return

        self.logger.info(f"Processing embeddings for {len(pages)} pages")

        try:
            async with TEIEmbeddingsClient(
                base_url=self.get_config_value("tei_endpoint", self.settings.tei_url),
                timeout_s=self.get_config_value(
                    "tei_timeout_s", self.settings.tei_timeout
                ),
            ) as tei_client:
                # Prepare texts for embedding
                texts = [page.content for page in pages]

                # Process in batches
                batches = pack_texts_into_batches(
                    texts,
                    max_items=self.get_config_value(
                        "tei_batch_size", self.settings.tei_batch_size
                    ),
                    target_chars=self.get_config_value(
                        "tei_target_chars_per_batch", 8000
                    ),
                )

                # Create embeddings map to correctly assign embeddings to pages
                embeddings_map = {}
                for batch in batches:
                    # Extract texts from (index, text) tuples
                    texts_only = [text for _, text in batch]
                    batch_embeddings = await tei_client.embed_texts(texts_only)

                    # Map embeddings back to original page indices
                    for (index, _), embedding in zip(
                        batch, batch_embeddings, strict=False
                    ):
                        embeddings_map[index] = embedding

                # Store embeddings in pages using correct indices
                for i, page in enumerate(pages):
                    page.embedding = embeddings_map.get(i)

                self.logger.info(
                    f"Generated embeddings for {len(embeddings_map)} pages"
                )

        except Exception as e:
            self.logger.error(f"Embeddings processing failed: {e}")

    async def _process_qdrant_upserts(self, pages: list[PageContent]) -> None:
        """Process Qdrant upserts for crawled pages."""
        if not self.get_config_value(
            "qdrant_url", self.settings.qdrant_url
        ) or not self.get_config_value("enable_qdrant", False):
            return

        self.logger.info(f"Upserting {len(pages)} pages to Qdrant")

        try:
            # Ensure collection exists on the same endpoint as the upsert
            async with QdrantClient(
                base_url=self.get_config_value("qdrant_url", self.settings.qdrant_url),
                api_key=self.get_config_value(
                    "qdrant_api_key", self.settings.qdrant_api_key
                ),
                timeout_s=15.0,
            ) as qdrant_client:
                await qdrant_client.ensure_collection(
                    name=self.get_config_value(
                        "qdrant_collection", self.settings.qdrant_collection
                    ),
                    size=self.get_config_value(
                        "qdrant_vector_size", self.settings.qdrant_vector_size
                    ),
                    vectors_name=self.get_config_value(
                        "qdrant_vectors_name", self.settings.qdrant_vectors_name
                    ),
                )
                # Prepare points for upsert
                points = []
                for page in pages:
                    if page.embedding:
                        # Use BLAKE2b hash converted to UUID for deterministic IDs (digest_size=16 for UUID compatibility)
                        blake2_hash = hashlib.blake2b(
                            page.url.encode("utf-8"), digest_size=16
                        ).digest()
                        # Convert BLAKE2b bytes directly to UUID (digest_size=16 produces exactly 16 bytes)
                        url_uuid = str(uuid.UUID(bytes=blake2_hash))
                        point: dict[str, Any] = {"id": url_uuid}
                        if self.get_config_value("qdrant_vectors_name"):
                            point["vectors"] = {
                                self.get_config_value(
                                    "qdrant_vectors_name"
                                ): page.embedding
                            }
                        else:
                            point["vector"] = page.embedding
                        point["payload"] = {
                            "url": page.url,
                            "title": page.title,
                            "text": page.content[:10000],  # Limit payload size
                            "word_count": page.word_count,
                            "crawl_timestamp": datetime.now(UTC).isoformat(),
                        }
                        points.append(point)

                if points:
                    await qdrant_client.upsert(
                        self.get_config_value(
                            "qdrant_collection", self.settings.qdrant_collection
                        ),
                        points,
                        wait=self.get_config_value(
                            "qdrant_upsert_wait", self.settings.qdrant_upsert_wait
                        ),
                    )
                    self.logger.info(f"Upserted {len(points)} points to Qdrant")

        except Exception as e:
            self.logger.error(f"Qdrant upsert failed: {e}")

    def get_last_pages(self) -> list[PageContent]:
        """Get the last crawled pages."""
        return self._last_pages

    def _get_optimized_browser_config(self, url: str) -> BrowserConfig:
        """Get optimized browser configuration based on URL and settings."""
        if hasattr(self.browser_factory, "get_config_for_url"):
            return self.browser_factory.get_config_for_url(url)
        return self.browser_factory.get_recommended_config()

    def _create_optimized_crawler_config(
        self, deep_crawl_strategy, url: str, stream: bool = False
    ) -> CrawlerRunConfig:
        """Create optimized CrawlerRunConfig with performance settings."""

        # Determine cache strategy
        cache_mode = self._get_cache_mode()

        # Build excluded selectors
        excluded_selectors = list(
            self.get_config_value(
                "excluded_selectors", self.settings.excluded_selectors
            )
            or []
        )

        # Build excluded tags
        excluded_tags = list(
            self.get_config_value("excluded_tags", self.settings.excluded_tags) or []
        )

        # Create content filter and markdown generator if enabled
        markdown_generator = None
        if (
            self.get_config_value("enable_content_filter", True)
            and self.get_config_value("content_filter_type", "pruning") != "none"
        ):
            content_filter = self._create_content_filter()
            markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)
            self.logger.info(
                f"Created {self.get_config_value('content_filter_type', 'pruning')} content filter for cleaner markdown"
            )

        # Create optimized configuration using only valid CrawlerRunConfig parameters
        crawler_config = {
            "deep_crawl_strategy": deep_crawl_strategy,
            "stream": stream,
            "cache_mode": cache_mode,
            "page_timeout": self.get_config_value("page_timeout", 30000),
            "word_count_threshold": max(
                1, self.get_config_value("min_word_count", 5)
            ),  # Reduced from 10 to 5
            "excluded_tags": excluded_tags,
            "excluded_selector": ", ".join(excluded_selectors)
            if excluded_selectors
            else None,
            # Content scoping (per docs)
            "css_selector": self.get_config_value("css_selector", None),
            "target_elements": self.get_config_value("target_elements", None),
            "wait_until": self._serialize_enum_value(
                self.get_config_value("wait_condition", "domcontentloaded")
            ),
            "delay_before_return_html": self.get_config_value(
                "html_delay_seconds", 2.0
            ),
            "only_text": self.get_config_value("enable_text_only_mode", False),
            "exclude_external_links": self.get_config_value(
                "exclude_external_links", False
            ),
            "exclude_social_media_links": self.get_config_value(
                "exclude_social_media_links", True
            ),
            "exclude_domains": self.get_config_value("exclude_domains", []),
            "exclude_social_media_domains": self.get_config_value(
                "exclude_social_media_domains", []
            ),
            "remove_forms": self.get_config_value("remove_forms", False),
            "exclude_external_images": self.get_config_value(
                "exclude_external_images", False
            ),
            "process_iframes": self.get_config_value("process_iframes", False),
            # Align with docs: respect robots.txt when configured
            "check_robots_txt": bool(
                self.get_config_value("respect_robots_txt", False)
            ),
            "semaphore_count": self.get_config_value("crawl_semaphore_count", 5),
            "mean_delay": self.get_config_value("mean_request_delay", 1.0),
            "max_range": self.get_config_value("max_request_delay_range", 2.0),
            "markdown_generator": markdown_generator,
        }

        # Remove None values
        crawler_config = {k: v for k, v in crawler_config.items() if v is not None}

        return CrawlerRunConfig(**crawler_config)

    def _get_cache_mode(self) -> CacheMode:
        """Get appropriate cache mode based on configuration."""
        strategy = self.get_config_value("cache_strategy", self.settings.cache_strategy)
        key = strategy.value if hasattr(strategy, "value") else str(strategy).lower()
        return self._CACHE_MODES.get(key, CacheMode.ENABLED)

    def _create_content_filter(self) -> PruningContentFilter | BM25ContentFilter:
        """Create content filter based on configuration."""
        if self.get_config_value("content_filter_type", "pruning") == "pruning":
            return PruningContentFilter(
                threshold=self.get_config_value("pruning_threshold", 0.45),
                threshold_type=self.get_config_value("pruning_threshold_type", "fixed"),
                min_word_threshold=self.get_config_value("pruning_min_words", 5),
            )
        elif self.get_config_value("content_filter_type", "pruning") == "bm25":
            if not self.get_config_value("bm25_user_query"):
                self.logger.warning(
                    "BM25 content filter selected but no user query configured, falling back to pruning filter"
                )
                return PruningContentFilter(
                    threshold=self.get_config_value("pruning_threshold", 0.45),
                    threshold_type=self.get_config_value(
                        "pruning_threshold_type", "fixed"
                    ),
                    min_word_threshold=self.get_config_value("pruning_min_words", 5),
                )
            return BM25ContentFilter(
                user_query=self.get_config_value("bm25_user_query"),
                bm25_threshold=self.get_config_value("bm25_threshold", 0.5),
            )
        else:
            # Fallback to pruning if invalid type
            return PruningContentFilter(
                threshold=self.get_config_value("pruning_threshold", 0.45),
                threshold_type=self.get_config_value("pruning_threshold_type", "fixed"),
                min_word_threshold=self.get_config_value("pruning_min_words", 5),
            )

    def get_performance_report(self) -> dict[str, Any]:
        """Get performance metrics."""
        duration = 0.0
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = self.stats["end_time"] - self.stats["start_time"]

        pages_per_second = self.stats["pages_crawled"] / duration if duration > 0 else 0

        return {
            "summary": {
                "pages_crawled": self.stats["pages_crawled"],
                "pages_failed": self.stats["pages_failed"],
                "success_rate": (
                    self.stats["pages_crawled"]
                    / max(1, self.stats["pages_crawled"] + self.stats["pages_failed"])
                ),
                "pages_per_second": pages_per_second,
                "total_duration": duration,
            },
            "system_performance": {
                "peak_memory_mb": 0,
                "average_cpu_usage": 0,
                "concurrent_sessions_peak": 1,
            },
            "error_analysis": {
                "total_errors": self.stats["pages_failed"],
                "error_rate": (
                    self.stats["pages_failed"]
                    / max(1, self.stats["pages_crawled"] + self.stats["pages_failed"])
                ),
            },
            "performance_config": {
                "cache_strategy": self._serialize_enum_value(
                    self.get_config_value("cache_strategy", "enabled")
                ),
                "wait_condition": self._serialize_enum_value(
                    self.get_config_value("wait_condition", "domcontentloaded")
                ),
                "html_delay_seconds": self.get_config_value("html_delay_seconds", 2.0),
                "text_only_mode": self.get_config_value("enable_text_only_mode", False),
                "semaphore_count": self.get_config_value("crawl_semaphore_count", 5),
                "browser_mode": self._serialize_enum_value(
                    self.get_config_value("browser_mode", "headless")
                ),
                "url_optimization": self.get_config_value(
                    "enable_url_based_optimization", False
                ),
            },
        }
