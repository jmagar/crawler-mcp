"""
Main orchestrator for optimized high-performance web crawler.

This module implements the main crawler orchestrator that uses Crawl4AI's
deep crawling APIs to discover and crawl multiple pages efficiently.
"""

import logging
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

from crawler_mcp.clients.qdrant_http_client import QdrantClient
from crawler_mcp.clients.tei_client import TEIEmbeddingsClient
from crawler_mcp.crawl_core.batch_utils import pack_texts_into_batches
from crawler_mcp.factories.browser_factory import BrowserFactory
from crawler_mcp.factories.content_extractor import ContentExtractorFactory
from crawler_mcp.models.crawl import PageContent
from crawler_mcp.models.responses import OptimizedCrawlResponse
from crawler_mcp.optimized_config import OptimizedConfig
from crawler_mcp.processing.result_converter import ResultConverter
from crawler_mcp.utils.monitoring import PerformanceMonitor


class CrawlOrchestrator:
    """
    High-performance web crawler orchestrator using Crawl4AI's deep crawling APIs.

    This orchestrator uses Crawl4AI's documented BFSDeepCrawlStrategy to handle
    both URL discovery and parallel crawling in a single unified process.
    """

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize the crawler orchestrator.

        Args:
            config: Optional configuration object
        """
        self.config = config or OptimizedConfig()
        self.logger = logging.getLogger(__name__)

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
                self.logger.debug(f"Hook {event} failed: {e}")

    async def crawl(
        self,
        start_url: str,
        max_urls: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> OptimizedCrawlResponse:
        """
        Crawl a website using Crawl4AI's deep crawling strategy.

        Args:
            start_url: Starting URL to crawl
            max_urls: Maximum number of pages to crawl
            stream: Whether to stream results
            **kwargs: Additional crawling parameters

        Returns:
            OptimizedCrawlResponse with crawl results
        """
        self.stats["start_time"] = time.time()
        max_urls = max_urls or self.config.max_urls_to_discover

        self.logger.info(f"Starting deep crawl of {start_url} (max: {max_urls} pages)")

        # Create domain-based URL filter
        domain = urlparse(start_url).netloc
        self.logger.info(f"Creating URL filter for domain: {domain}")
        url_filter = URLPatternFilter(
            patterns=[f"https://{domain}/*", f"http://{domain}/*"]
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

        # Create deep crawl strategy
        self.logger.info(
            f"Creating BFS strategy: max_depth=3, max_pages={max_urls}, threshold={self.config.url_score_threshold}"
        )
        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=3,
            include_external=False,
            max_pages=max_urls,
            filter_chain=FilterChain([url_filter]),
            url_scorer=scorer,
            score_threshold=0.0,  # Lower threshold to allow more pages
        )

        # Configure crawler
        browser_config = self.browser_factory.get_recommended_config()

        # Add deep crawl strategy to config
        crawler_config_dict = {
            "deep_crawl_strategy": deep_crawl_strategy,
            "stream": stream,
            "page_timeout": self.config.page_timeout,
            "word_count_threshold": self.config.min_word_count,
            "excluded_tags": ["script", "style", "noscript"],
        }

        # Create config object
        config = CrawlerRunConfig(**crawler_config_dict)

        # Execute crawl
        try:
            self.logger.info("Executing deep crawl with BFSDeepCrawlStrategy")

            async with AsyncWebCrawler(config=browser_config) as crawler:
                if stream:
                    # Stream mode - process results as they arrive
                    crawl_results = []
                    async for result in await crawler.arun(start_url, config=config):
                        crawl_results.append(result)
                        self._trigger_hook(
                            "page_crawled",
                            url=result.url,
                            content_length=len(result.markdown or ""),
                            crawl_time=0.0,
                        )
                        self.stats["pages_crawled"] += 1
                else:
                    # Non-stream mode - get all results at once
                    crawl_results = await crawler.arun(start_url, config=config)
                    if not isinstance(crawl_results, list):
                        crawl_results = [crawl_results]

                    self.stats["pages_crawled"] = len(
                        [r for r in crawl_results if r.success]
                    )
                    self.stats["pages_failed"] = len(
                        [r for r in crawl_results if not r.success]
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
            )

        except Exception as e:
            self.logger.error(f"Deep crawl failed: {e}", exc_info=True)
            self.stats["end_time"] = time.time()

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
                    max_batch_size=self.config.tei_batch_size,
                    max_chars_per_item=self.config.tei_max_input_chars,
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
            async with QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout_s=15.0,
            ) as qdrant_client:
                # Prepare points for upsert
                points = []
                for page in pages:
                    if page.embedding:
                        point = {
                            "id": page.url,
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
                    await qdrant_client.upsert_points(
                        collection_name=self.config.qdrant_collection,
                        points=points,
                        wait=self.config.qdrant_upsert_wait,
                    )
                    self.logger.info(f"Upserted {len(points)} points to Qdrant")

        except Exception as e:
            self.logger.error(f"Qdrant upsert failed: {e}")

    def get_last_pages(self) -> list[PageContent]:
        """Get the last crawled pages."""
        return self._last_pages

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
        }
