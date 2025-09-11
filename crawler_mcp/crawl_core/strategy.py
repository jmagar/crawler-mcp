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
from datetime import datetime
from typing import Any, ClassVar
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from crawler_mcp.clients.qdrant_http_client import QdrantClient
from crawler_mcp.clients.tei_client import TEIEmbeddingsClient
from crawler_mcp.config import get_settings
from crawler_mcp.core.vectors.collections import CollectionManager
from crawler_mcp.crawl_core.batch_utils import pack_texts_into_batches
from crawler_mcp.factories.browser_factory import BrowserFactory
from crawler_mcp.factories.content_extractor import ContentExtractorFactory
from crawler_mcp.models.crawl import PageContent
from crawler_mcp.models.responses import OptimizedCrawlResponse
from crawler_mcp.optimized_config import OptimizedConfig
from crawler_mcp.processing.result_converter import ResultConverter
from crawler_mcp.utils.monitoring import PerformanceMonitor


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

    def apply(self, url: str) -> bool:
        """
        Return False to exclude URL, True to include.

        Args:
            url: URL to check against exclusion patterns

        Returns:
            False if URL should be excluded, True if it should be included
        """
        for pattern in self.patterns:
            if pattern.search(url):
                return False  # Exclude this URL
        return True  # Include if no exclusion pattern matches


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

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize the crawler orchestrator.

        Args:
            config: Optional configuration object
        """
        self.config = config or OptimizedConfig()
        self.logger = logging.getLogger(__name__)

        # Ensure exclusion patterns are available from settings
        if not hasattr(self.config, "crawl_exclude_url_patterns"):
            settings = get_settings()
            self.config.crawl_exclude_url_patterns = settings.crawl_exclude_url_patterns

        # Initialize components
        self.browser_factory = BrowserFactory(self.config)
        self.content_extractor = ContentExtractorFactory(self.config)
        self.result_converter = ResultConverter(self.config)
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
        max_urls = max_urls or self.config.max_urls_to_discover

        # Get streaming setting from main config (single source of truth)
        settings = get_settings()
        stream = settings.enable_streaming

        self.logger.info(f"Starting deep crawl of {start_url} (max: {max_urls} pages)")

        # Create domain-based URL filter
        domain = urlparse(start_url).netloc
        self.logger.info(f"Creating URL filter for domain: {domain}")
        url_filter = URLPatternFilter(
            patterns=[f"https://{domain}/*", f"http://{domain}/*"]
        )

        # Create exclusion filter using config patterns
        exclusion_patterns = getattr(self.config, "crawl_exclude_url_patterns", [])
        exclusion_filter = ExclusionFilter(exclusion_patterns)
        self.logger.info(
            f"Created exclusion filter with {len(exclusion_patterns)} patterns"
        )

        # Create relevance scorer with documentation keywords
        scorer = KeywordRelevanceScorer(
            keywords=[
                "documentation",
                "api",
                "guide",
                "tutorial",
                "docs",
                "reference",
                "manual",
                "help",
                "getting-started",
            ],
            weight=0.7,
        )

        # Create deep crawl strategy with both inclusion and exclusion filters
        self.logger.info(
            f"Creating BFS strategy: max_depth=3, max_pages={max_urls}, threshold={self.config.url_score_threshold}"
        )
        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=3,
            include_external=False,
            max_pages=max_urls,
            filter_chain=FilterChain([url_filter, exclusion_filter]),
            url_scorer=scorer,
            score_threshold=0.0,  # Lower threshold to allow more pages
        )

        # Configure crawler with performance optimizations
        browser_config = self._get_optimized_browser_config(start_url)
        config = self._create_optimized_crawler_config(
            deep_crawl_strategy=deep_crawl_strategy, stream=stream, url=start_url
        )

        # Execute crawl with enhanced logging
        try:
            self.logger.info("Executing deep crawl with BFSDeepCrawlStrategy")
            self.logger.info(
                f"Memory threshold: {self.config.memory_threshold_percent}%, Max sessions: {self.config.max_session_permit}"
            )
            self.logger.info(
                f"Semaphore count: {self.config.crawl_semaphore_count}, Delays: {self.config.mean_request_delay}s-{self.config.max_request_delay_range}s"
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                if stream:
                    # Stream mode - process results as they arrive with progress logging
                    crawl_results = []
                    page_count = 0
                    start_time = time.time()

                    self.logger.info("Starting streaming crawl...")
                    try:
                        async for result in await crawler.arun(
                            start_url, config=config
                        ):
                            # Validate result type to prevent float/invalid objects
                            if not hasattr(result, "url") or not hasattr(
                                result, "success"
                            ):
                                self.logger.debug(
                                    f"Skipping invalid result object: {type(result)}"
                                )
                                continue

                            page_count += 1
                            crawl_results.append(result)

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
                                self.logger.info(
                                    f"✅ Page {page_count}: {result.url} ({content_length} chars, {rate:.1f} pages/sec)"
                                )
                                self._trigger_hook(
                                    "page_crawled",
                                    url=result.url,
                                    content_length=content_length,
                                    crawl_time=elapsed,
                                )
                            else:
                                self.logger.warning(
                                    f"❌ Page {page_count}: {result.url} - Error: {result.error_message}"
                                )
                                self._trigger_hook(
                                    "page_failed",
                                    url=result.url,
                                    error=result.error_message,
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

                else:
                    # Non-stream mode - get all results at once
                    self.logger.info("Starting non-streaming crawl...")
                    crawl_results = await crawler.arun(start_url, config=config)
                    if not isinstance(crawl_results, list):
                        crawl_results = [crawl_results]

                    self.stats["pages_crawled"] = len(
                        [r for r in crawl_results if r.success]
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
            for result in crawl_results:
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
            if self.config.enable_embeddings and pages:
                await self._process_embeddings(pages)

            # Process Qdrant upserts if enabled
            if self.config.enable_qdrant and pages:
                await self._process_qdrant_upserts(pages)

            self.stats["end_time"] = time.time()

            # Calculate crawl duration and log final timing
            duration = self.stats["end_time"] - self.stats["start_time"]
            self.logger.info(
                f"Crawl completed in {duration:.2f} seconds - "
                f"{len(pages)} pages processed at {len(pages) / duration:.1f} pages/sec"
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
        if not self.config.tei_endpoint:
            self.logger.warning("TEI endpoint not configured, skipping embeddings")
            return

        self.logger.info(f"Processing embeddings for {len(pages)} pages")

        try:
            async with TEIEmbeddingsClient(
                base_url=self.config.tei_endpoint, timeout_s=self.config.tei_timeout_s
            ) as tei_client:
                # Prepare texts for embedding
                texts = [page.content for page in pages]

                # Process in batches
                batches = pack_texts_into_batches(
                    texts,
                    max_items=self.config.tei_batch_size,
                    target_chars=self.config.tei_target_chars_per_batch,
                )

                embeddings = []
                for batch in batches:
                    batch_embeddings = await tei_client.embed_texts(batch)
                    embeddings.extend(batch_embeddings)

                # Store embeddings in pages
                for page, embedding in zip(pages, embeddings, strict=False):
                    page.embedding = embedding

                self.logger.info(f"Generated embeddings for {len(embeddings)} pages")

        except Exception as e:
            self.logger.error(f"Embeddings processing failed: {e}")

    async def _process_qdrant_upserts(self, pages: list[PageContent]) -> None:
        """Process Qdrant upserts for crawled pages."""
        if not self.config.qdrant_url or not self.config.enable_qdrant:
            return

        self.logger.info(f"Upserting {len(pages)} pages to Qdrant")

        try:
            # Ensure collection exists before attempting upsert
            collection_manager = CollectionManager()
            await collection_manager.ensure_collection_exists()

            async with QdrantClient(
                base_url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout_s=15.0,
            ) as qdrant_client:
                # Prepare points for upsert
                points = []
                for page in pages:
                    if page.embedding:
                        # Use MD5 hash converted to UUID for deterministic IDs (MD5 is 16 bytes, perfect for UUID)
                        md5_hash = hashlib.md5(page.url.encode("utf-8")).digest()
                        # Convert MD5 bytes directly to UUID (MD5 produces exactly 16 bytes)
                        url_uuid = str(uuid.UUID(bytes=md5_hash))
                        point = {
                            "id": url_uuid,
                            "vector": page.embedding,
                            "payload": {
                                "url": page.url,
                                "title": page.title,
                                "text": page.content[:10000],  # Limit payload size
                                "word_count": page.word_count,
                                "crawl_timestamp": datetime.utcnow().isoformat(),
                            },
                        }
                        points.append(point)

                if points:
                    await qdrant_client.upsert(
                        self.config.qdrant_collection,
                        points,
                        wait=self.config.qdrant_upsert_wait,
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
        self, deep_crawl_strategy, stream: bool, url: str
    ) -> CrawlerRunConfig:
        """Create optimized CrawlerRunConfig with performance settings."""

        # Determine cache strategy
        cache_mode = self._get_cache_mode()

        # Build excluded selectors
        excluded_selectors = self.config.excluded_selectors.copy()

        # Build excluded tags
        excluded_tags = self.config.excluded_tags.copy()

        # Create content filter and markdown generator if enabled
        markdown_generator = None
        if (
            self.config.enable_content_filter
            and self.config.content_filter_type != "none"
        ):
            content_filter = self._create_content_filter()
            markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)
            self.logger.info(
                f"Created {self.config.content_filter_type} content filter for cleaner markdown"
            )

        # Create optimized configuration using only valid CrawlerRunConfig parameters
        crawler_config = {
            "deep_crawl_strategy": deep_crawl_strategy,
            "stream": stream,
            "cache_mode": cache_mode,
            "page_timeout": self.config.page_timeout,
            "word_count_threshold": self.config.min_word_count,
            "excluded_tags": excluded_tags,
            "excluded_selector": ", ".join(excluded_selectors)
            if excluded_selectors
            else None,
            "wait_until": self.config.wait_condition,
            "delay_before_return_html": self.config.html_delay_seconds,
            "only_text": self.config.enable_text_only_mode,
            "exclude_external_links": self.config.exclude_external_links,
            "remove_forms": self.config.remove_forms,
            "exclude_external_images": self.config.exclude_external_images,
            "semaphore_count": self.config.crawl_semaphore_count,
            "mean_delay": self.config.mean_request_delay,
            "max_range": self.config.max_request_delay_range,
            "markdown_generator": markdown_generator,
        }

        # Remove None values
        crawler_config = {k: v for k, v in crawler_config.items() if v is not None}

        return CrawlerRunConfig(**crawler_config)

    def _get_cache_mode(self) -> CacheMode:
        """Get appropriate cache mode based on configuration."""
        strategy = self.config.cache_strategy
        return self._CACHE_MODES.get(strategy, CacheMode.ENABLED)

    def _create_content_filter(self) -> PruningContentFilter | BM25ContentFilter:
        """Create content filter based on configuration."""
        if self.config.content_filter_type == "pruning":
            return PruningContentFilter(
                threshold=self.config.pruning_threshold,
                threshold_type=self.config.pruning_threshold_type,
                min_word_threshold=self.config.pruning_min_words,
            )
        elif self.config.content_filter_type == "bm25":
            if not self.config.bm25_user_query:
                self.logger.warning(
                    "BM25 content filter selected but no user query configured, falling back to pruning filter"
                )
                return PruningContentFilter(
                    threshold=self.config.pruning_threshold,
                    threshold_type=self.config.pruning_threshold_type,
                    min_word_threshold=self.config.pruning_min_words,
                )
            return BM25ContentFilter(
                user_query=self.config.bm25_user_query,
                bm25_threshold=self.config.bm25_threshold,
            )
        else:
            # Fallback to pruning if invalid type
            return PruningContentFilter(
                threshold=self.config.pruning_threshold,
                threshold_type=self.config.pruning_threshold_type,
                min_word_threshold=self.config.pruning_min_words,
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
                "cache_strategy": self.config.cache_strategy,
                "wait_condition": self.config.wait_condition,
                "html_delay_seconds": self.config.html_delay_seconds,
                "text_only_mode": self.config.enable_text_only_mode,
                "semaphore_count": self.config.crawl_semaphore_count,
                "browser_mode": self.config.browser_mode,
                "url_optimization": self.config.enable_url_based_optimization,
            },
        }
