"""
Optimized web crawling strategy with streaming and caching support.
"""

import contextlib
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from crawl4ai import (  # type: ignore
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from crawl4ai import CrawlResult as Crawl4aiResult  # type: ignore
from crawl4ai.deep_crawling import (  # type: ignore
    BFSDeepCrawlStrategy,
)
from crawl4ai.extraction_strategy import (  # type: ignore
    CosineStrategy,
    LLMExtractionStrategy,
)

from ..config import settings
from ..core.logging import get_logger
from ..core.utils import suppress_stdout
from ..models.crawl import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)

# Removed DefaultMarkdownGeneratorImpl import - using direct crawl4ai imports instead
from .base import BaseCrawlStrategy

logger = get_logger(__name__)


class WebCrawlStrategy(BaseCrawlStrategy):
    """
    High-performance web crawling strategy with streaming, caching, and GPU acceleration.
    Optimized for RTX 4070 + i7-13700K performance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.memory_manager = None
        self.markdown_generator = None
        self._initialize_markdown_generator()

    def _initialize_markdown_generator(self) -> None:
        """Initialize reusable markdown generator with enhanced error handling and fallbacks."""
        try:
            from crawl4ai import DefaultMarkdownGenerator  # type: ignore
            from crawl4ai.content_filter_strategy import (
                PruningContentFilter,  # type: ignore
            )

            # Create content filter with conservative settings that match working single-page config
            content_filter = PruningContentFilter(
                threshold=getattr(
                    settings, "crawl_pruning_threshold", 0.25
                ),  # Conservative threshold like single-page
                threshold_type="dynamic",  # Dynamic scoring like working single-page
                min_word_threshold=getattr(
                    settings, "crawl_min_word_threshold", 3
                ),  # Keep short text blocks
            )

            # Create reusable markdown generator (match orchestrator config)
            self.markdown_generator = DefaultMarkdownGenerator(
                content_filter=content_filter
                # Remove content_source="cleaned_html" - might be causing hash placeholders
            )
            self.logger.info(
                f"Successfully initialized markdown generator: {type(self.markdown_generator)}"
            )

        except ImportError as e:
            self.logger.error(f"DefaultMarkdownGenerator not available: {e}")
            # Try basic fallback without content filter
            try:
                from crawl4ai import DefaultMarkdownGenerator  # type: ignore

                self.markdown_generator = DefaultMarkdownGenerator()
                self.logger.warning(
                    "Using basic DefaultMarkdownGenerator fallback without content filter"
                )
            except Exception as e2:
                self.logger.error(
                    f"Basic DefaultMarkdownGenerator fallback failed: {e2}"
                )
                self.markdown_generator = None

        except Exception as e:
            self.logger.error(
                f"Unexpected error initializing markdown generator with content filter: {e}"
            )
            # Try creating generator without content filter as fallback
            try:
                from crawl4ai import DefaultMarkdownGenerator  # type: ignore

                self.markdown_generator = DefaultMarkdownGenerator()
                self.logger.warning(
                    f"Created fallback markdown generator without content filter due to error: {e}"
                )
            except Exception as e2:
                self.logger.error(
                    f"All markdown generator initialization attempts failed: {e2}"
                )
                self.markdown_generator = None

        # Final validation
        if self.markdown_generator is None:
            self.logger.error(
                "CRITICAL: No markdown generator available - batch processing will likely produce hash placeholders!"
            )

    def _get_browser_args(self) -> list[str]:
        """Get optimized browser arguments for performance and containerization."""
        args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            # Disable GPU for text scraping performance (GPU adds overhead for DOM parsing)
            "--disable-gpu",  # Explicitly disable GPU acceleration
            "--disable-software-rasterizer",  # Disable software fallback rasterizer
            "--disable-gpu-sandbox",  # Disable GPU sandbox
            # CPU-optimized settings for text extraction
            "--renderer-process-limit=4",  # Match 4 browser pool size
            "--max-renderer-processes=4",  # Conservative renderer limit
            # Essential performance flags only
            "--disable-extensions",  # Disable extensions for faster startup
            "--disable-plugins",  # Disable plugins for security and speed
            "--disable-web-security",  # Allow cross-origin requests for scraping
            "--disable-background-timer-throttling",  # Faster rendering
            "--max_old_space_size=4096",  # Conservative V8 memory limit for efficiency
        ]

        # Conditional arguments based on settings
        if getattr(settings, "crawl_block_images", False):
            args.append("--disable-images")
        if getattr(settings, "crawl_disable_js", False):
            args.append("--disable-javascript")

        return args

    async def _initialize_managers(self) -> None:
        """Initialize required managers."""
        if not self.memory_manager:
            from ..core.memory import get_memory_manager

            self.memory_manager = get_memory_manager()

    async def validate_request(self, request: CrawlRequest) -> bool:
        """Validate web crawl request."""
        if not request.url:
            self.logger.error("URL is required for web crawling")
            return False

        if request.max_pages is not None and (
            request.max_pages < 1 or request.max_pages > 2000
        ):
            self.logger.error(
                "max_pages must be between 1 and 2000, got %s", request.max_pages
            )
            return False

        if request.max_depth is not None and (
            request.max_depth < 1 or request.max_depth > 5
        ):
            self.logger.error(
                "max_depth must be between 1 and 5, got %s", request.max_depth
            )
            return False

        return True

    async def execute(
        self,
        request: CrawlRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """
        Execute optimized web crawling by delegating to the crawl4ai library.
        """

        self.logger.info("WebCrawlStrategy.execute() started for URLs: %s", request.url)
        await self._initialize_managers()

        start_time = time.time()
        self.logger.info(
            "Starting web crawl: %s (max_pages: %s, max_depth: %s)",
            request.url,
            request.max_pages,
            request.max_depth,
        )

        await self.pre_execute_setup()

        try:
            if self.memory_manager is None:
                raise RuntimeError("Memory manager not initialized")
            if not await self.memory_manager.can_handle_crawl(request.max_pages or 100):
                self.logger.warning(
                    "System may have insufficient memory for crawl, proceeding with caution"
                )

            # Optimized browser config for high-performance crawling
            browser_config = BrowserConfig(
                headless=settings.crawl_headless,
                browser_type=settings.crawl_browser,
                browser_mode="builtin",  # CRITICAL: Use builtin browser for performance
                viewport_width=1200,  # Consistent viewport for better content extraction
                viewport_height=800,
                light_mode=True,  # Optimized performance mode
                text_mode=True,  # Disable images for faster crawling
                verbose=False,  # Suppress Crawl4AI console output for MCP compatibility
                # Performance-optimized flags for containerized environments
                extra_args=self._get_browser_args(),
            )

            # Create fresh AsyncWebCrawler instance with stdout suppressed
            with suppress_stdout():
                browser = AsyncWebCrawler(config=browser_config)
                await browser.start()

            # Sitemap preseeding: discover and parse sitemap URLs to bias deep crawling
            # Discover ALL seeds from sitemap (don't limit by max_pages during discovery)
            sitemap_seeds: list[str] = []
            for u in request.url:
                sitemap_seeds.extend(
                    await self._discover_sitemap_seeds(
                        u, limit=10000
                    )  # High limit to get all sitemap URLs
                )
            # Deduplicate while preserving order
            seen = set()
            deduplicated = []
            for s in sitemap_seeds:
                if s not in seen:
                    seen.add(s)
                    deduplicated.append(s)
            sitemap_seeds = deduplicated
            self.logger.info(
                "Discovered %s sitemap seeds: %s...",
                len(sitemap_seeds),
                sitemap_seeds[:5],
            )

            run_config = await self._build_run_config(request, sitemap_seeds)
            self.logger.info(
                "Deep crawl strategy configured: %s",
                type(run_config.deep_crawl_strategy).__name__
                if run_config.deep_crawl_strategy
                else "None",
            )
            self.logger.info(f"DEBUG: run_config.stream = {run_config.stream}")
            self.logger.info(
                f"DEBUG: deep_crawl_strategy is None: {run_config.deep_crawl_strategy is None}"
            )
            if run_config.deep_crawl_strategy:
                self.logger.info(
                    "Max pages: %s, Max depth: %s",
                    getattr(run_config.deep_crawl_strategy, "max_pages", "Unknown"),
                    getattr(run_config.deep_crawl_strategy, "max_depth", "Unknown"),
                )

            pages: list[dict[str, Any]] = []
            errors: list[str] = []
            total_bytes = 0
            unique_domains = set()
            total_links_discovered = 0
            max_pages = request.max_pages or 100

            if progress_callback:
                progress_callback(0, max_pages, "Starting crawl...")

            # PERFORMANCE FIX: Use concurrent arun for batch processing with proper content extraction
            crawl_count = 0
            if max_pages > 1 and sitemap_seeds:
                self.logger.info(
                    f"Using concurrent arun with {len(sitemap_seeds)} URLs for optimal performance"
                )
                successful_results, errors = await self._crawl_using_concurrent_arun(
                    browser,
                    sitemap_seeds[:max_pages],
                    run_config,
                    request,
                    progress_callback,
                )
            elif max_pages > 1:
                # No sitemap seeds, use the main URL with concurrent processing
                self.logger.info(
                    "Using concurrent arun for multiple URLs (no sitemap seeds)"
                )
                successful_results, errors = await self._crawl_using_concurrent_arun(
                    browser,
                    request.url[:max_pages],
                    run_config,
                    request,
                    progress_callback,
                )
            else:
                # Single page - use simple arun for consistency
                self.logger.info("Using simple arun for single page")
                try:
                    result = await browser.arun(url=request.url[0], config=run_config)
                    if result.success:
                        successful_results = [result]
                        errors = []
                    else:
                        successful_results = []
                        errors = [
                            f"Failed to crawl {result.url}: {result.error_message}"
                        ]
                except Exception as e:
                    successful_results = []
                    errors = [f"Exception crawling {request.url[0]}: {e}"]

            # Process crawling results
            pages = []

            # Process results outside suppress_stdout context for debugging
            # Prefer fit_markdown for cleaner content, respecting global setting when request doesn't specify
            prefer_fit_markdown = bool(
                request.prefer_fit_markdown
                if getattr(request, "prefer_fit_markdown", None) is not None
                else getattr(settings, "crawl_prefer_fit_markdown", True)
            )
            for result in successful_results:
                self.logger.info("Processing successful result for %s", result.url)
                page_content = self._to_page_content(result, prefer_fit_markdown)
                self.logger.info(
                    "Created PageContent with %s words for %s",
                    page_content.word_count,
                    result.url,
                )
                pages.append(page_content)
                total_bytes += len(page_content.content)
                unique_domains.add(urlparse(page_content.url).netloc)
                total_links_discovered += len(page_content.links)

                if (
                    self.memory_manager
                    and await self.memory_manager.check_memory_pressure()
                ):
                    self.logger.warning(
                        "Memory pressure detected during crawl, may slow down"
                    )

                if progress_callback:
                    progress_callback(
                        len(pages),
                        max_pages,
                        f"Crawled: {page_content.url[:60]}...",
                    )

            self.logger.info(
                "Crawl loop completed: %s results processed, %s successful pages",
                crawl_count,
                len(pages),
            )
            end_time = time.time()
            crawl_duration = end_time - start_time
            pages_per_second = len(pages) / crawl_duration if crawl_duration > 0 else 0
            avg_page_size = total_bytes / len(pages) if pages else 0

            statistics = CrawlStatistics(
                total_pages_requested=max_pages,
                total_pages_crawled=len(pages),
                total_pages_failed=len(errors),
                unique_domains=len(unique_domains),
                total_links_discovered=total_links_discovered,
                total_bytes_downloaded=total_bytes,
                crawl_duration_seconds=crawl_duration,
                pages_per_second=pages_per_second,
                average_page_size=avg_page_size,
            )

            crawl_result = CrawlResult(
                request_id=f"web_crawl_{int(time.time())}",
                status=CrawlStatus.COMPLETED,
                urls=request.url,  # Already converted to list by validator
                pages=pages,
                errors=errors,
                statistics=statistics,
            )

            self.logger.info(
                "Web crawl completed: %s pages, %.1fs, %.1f pages/sec, %s errors",
                len(pages),
                crawl_duration,
                pages_per_second,
                len(errors),
            )

            return crawl_result

        except Exception as e:
            self.logger.error("Web crawl failed: %s", e, exc_info=True)
            return CrawlResult(
                request_id=f"web_crawl_failed_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=request.url,  # Already converted to list by validator
                pages=[],
                errors=[str(e)],
                statistics=CrawlStatistics(),
            )

        finally:
            # Clean up browser instance
            if "browser" in locals():
                try:
                    with suppress_stdout():
                        await browser.close()
                except Exception as e:
                    self.logger.warning("Error closing browser: %s", e)

            await self.post_execute_cleanup()

    def _extract_text_from_html(self, html: str | None) -> str:
        """Extract plain text from HTML as a final fallback."""
        if not html:
            return ""

        try:
            # Simple HTML tag removal (basic fallback)
            import re

            # Remove script and style tags and their content
            html = re.sub(
                r"<(script|style)[^>]*>.*?</\1>",
                "",
                html,
                flags=re.DOTALL | re.IGNORECASE,
            )
            # Remove all other HTML tags
            text = re.sub(r"<[^>]+>", "", html)
            # Clean up whitespace
            text = re.sub(r"\s+", " ", str(text))
            text = self._safe_strip(text)
            return text
        except Exception as e:
            self.logger.warning("Failed to extract text from HTML: %s", e)
            return ""

    def _safe_strip(self, content: Any) -> str:
        """Safely strip content, handling lists and other types."""
        try:
            if content is None:
                return ""
            if isinstance(content, list):
                joined = "\n".join(str(item) for item in content)
                return joined.strip() if hasattr(joined, "strip") else str(joined)
            content_str = str(content)
            return content_str.strip() if hasattr(content_str, "strip") else content_str
        except Exception as e:
            self.logger.error(
                f"Safe strip failed on {type(content)} with value {content!r}: {e}",
                exc_info=True,
            )
            try:
                return str(content) if content else ""
            except Exception:
                return ""

    def _extract_markdown(
        self, result: Crawl4aiResult, prefer_fit_markdown: bool = True
    ) -> str:
        """Extract markdown content from crawl4ai result using best practices."""
        if not result.success or not hasattr(result, "markdown") or not result.markdown:
            self.logger.warning(f"No markdown content available for {result.url}")
            return ""

        try:
            markdown_obj = result.markdown

            # Extract from MarkdownGenerationResult object
            if (
                prefer_fit_markdown
                and hasattr(markdown_obj, "fit_markdown")
                and markdown_obj.fit_markdown
            ):
                content = str(markdown_obj.fit_markdown).strip()
                if len(content) > 10:
                    return content

            if hasattr(markdown_obj, "raw_markdown") and markdown_obj.raw_markdown:
                content = str(markdown_obj.raw_markdown).strip()
                if len(content) > 10:
                    return content

            # Fallback for direct string content
            if isinstance(markdown_obj, str):
                content = markdown_obj.strip()
                if content and len(content) > 10:
                    return content

            self.logger.debug(f"No usable markdown content for {result.url}")
            return ""

        except Exception as e:
            self.logger.debug(f"Error extracting markdown from {result.url}: {e}")
            return ""

    def _minimal_content_cleanup(self, content: str) -> str:
        """Apply minimal content cleanup - let crawl4ai's content filtering do the heavy lifting."""
        if not content:
            return ""

        # Convert to string and strip safely
        content = self._safe_strip(content)

        # Only basic cleanup - crawl4ai's PruningContentFilter should handle most issues
        if getattr(settings, "clean_ui_artifacts", True):
            # Remove excessive whitespace only
            content = re.sub(r"\n{3,}", "\n\n", content)
            content = re.sub(r"\s{2,}", " ", content)

        return content.strip()

    def _to_page_content(
        self, result: Crawl4aiResult, prefer_fit_markdown: bool = True
    ) -> PageContent:
        """Converts a crawl4ai result to a PageContent object."""
        try:
            return self._to_page_content_impl(result, prefer_fit_markdown)
        except AttributeError as e:
            if "'list' object has no attribute 'strip'" in str(e):
                self.logger.warning(
                    f"Strip error caught for {result.url}, using fallback content extraction"
                )
                # Fallback: use our enhanced extraction method
                try:
                    self.logger.info(
                        "Using enhanced fallback extraction for %s", result.url
                    )
                    # Use simplified markdown extraction
                    markdown_content = self._extract_markdown(
                        result, prefer_fit_markdown
                    )

                    # If that still fails, try additional fallbacks
                    if not markdown_content or len(markdown_content.split()) <= 1:
                        self.logger.warning(
                            "Enhanced fallback returned minimal content, trying text extraction for %s",
                            result.url,
                        )
                        if hasattr(result, "text") and result.text:
                            text_content = self._safe_strip(result.text)
                            if text_content and len(text_content) > 10:
                                markdown_content = text_content
                        elif hasattr(result, "cleaned_text") and result.cleaned_text:
                            cleaned_content = self._safe_strip(result.cleaned_text)
                            if cleaned_content and len(cleaned_content) > 10:
                                markdown_content = cleaned_content

                    # Calculate word count safely
                    word_count = (
                        len(markdown_content.split()) if markdown_content else 0
                    )

                    return PageContent(
                        url=getattr(result, "url", ""),
                        content=markdown_content,  # This is the required field
                        markdown=markdown_content,  # Keep this for compatibility
                        title=getattr(result, "title", "") or "",
                        word_count=word_count,
                        links_count=0,
                        images_count=0,
                        metadata=getattr(result, "metadata", {}) or {},
                    )
                except Exception as inner_e:
                    self.logger.error(
                        f"Fallback extraction also failed for {result.url}: {inner_e}"
                    )
                    return PageContent(
                        url=getattr(result, "url", ""),
                        content="",  # Required field
                        markdown="",
                        title=getattr(result, "title", "") or "",
                        word_count=0,
                        links_count=0,
                        images_count=0,
                        metadata=getattr(result, "metadata", {}) or {},
                    )
            else:
                raise
        except Exception as e:
            self.logger.error(
                f"Exception in _to_page_content for {result.url}: {e}", exc_info=True
            )
            # Return empty page content on error
            return PageContent(
                url=getattr(result, "url", ""),
                content="",  # Required field
                markdown="",
                title=getattr(result, "title", "") or "",
                word_count=0,
                links_count=0,
                images_count=0,
                metadata=getattr(result, "metadata", {}) or {},
            )

    def _to_page_content_impl(
        self, result: Crawl4aiResult, prefer_fit_markdown: bool = True
    ) -> PageContent:
        """Convert crawl4ai result to PageContent using simplified, optimized approach."""

        # CRITICAL FIX: Check if content was pre-extracted by orchestrator approach
        if hasattr(result, "_orchestrator_extracted_content"):
            # Use the pre-extracted content from the working orchestrator approach
            content = result._orchestrator_extracted_content
            word_count = result._orchestrator_word_count
            self.logger.debug(
                f"Using pre-extracted orchestrator content: {word_count} words for {result.url}"
            )
        else:
            # Fall back to the old extraction method (still broken for BFS)
            content = self._extract_markdown(result, prefer_fit_markdown)

            # Apply minimal cleanup - let crawl4ai do the heavy lifting
            content = self._minimal_content_cleanup(content)

            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content) if content else ""

            # Calculate word count efficiently
            word_count = len(content.split()) if content else 0

        return PageContent(
            url=result.url,
            title=result.metadata.get("title", "") if result.metadata else "",
            content=content,
            html=result.html if hasattr(result, "html") else "",
            markdown=content,  # Use the same content as markdown
            word_count=word_count,
            links=[
                link.get("href", link) if isinstance(link, dict) else str(link)
                for link in (result.links.get("internal", []) if result.links else [])
            ],
            images=[
                img.get("src", img) if isinstance(img, dict) else str(img)
                for img in (result.media.get("images", []) if result.media else [])
            ],
            metadata={
                "depth": result.metadata.get("depth", 0) if result.metadata else 0,
                "status_code": getattr(result, "status_code", 200),
                "response_headers": dict(result.response_headers or {}),
                "extraction_method": "crawl4ai_optimized",
            },
            timestamp=datetime.fromtimestamp(time.time()),
        )

    async def _build_run_config(
        self, request: CrawlRequest, sitemap_seeds: list[str] | None = None
    ) -> CrawlerRunConfig:
        """Build Crawl4AI CrawlerRunConfig aligned with deep crawling and streaming."""
        # Cache mode mapping - Performance optimized
        # With our optimized markdown_generator, we can safely use caching for performance
        # Fall back to BYPASS if caching is disabled in settings
        cache_mode = (
            CacheMode.ENABLED
            if getattr(settings, "crawl_enable_caching", True)
            else CacheMode.BYPASS
        )

        # Timeout: Crawl4AI expects milliseconds; coerce if likely in seconds
        timeout_val = getattr(settings, "crawler_timeout", 30000)
        page_timeout = (
            int(timeout_val * 1000)
            if isinstance(timeout_val, int | float) and timeout_val < 1000
            else int(timeout_val)
        )

        # Deep crawl strategy (enable for multi-page or multi-depth crawls)
        deep_strategy = None
        max_pages = (
            request.max_pages
            if request.max_pages is not None
            else getattr(settings, "crawl_max_pages", 100)
        )
        max_depth = (
            request.max_depth
            if request.max_depth is not None
            else getattr(settings, "crawl_max_depth", 3)
        )

        # CRITICAL FIX: Disable BFS deep crawling due to hash placeholder bug in crawl4ai 0.7.4
        # Use sequential single-page approach instead which works correctly
        deep_strategy = None
        self.logger.info(
            f"BFS deep crawling disabled due to hash placeholder bug - using sequential approach: max_pages={max_pages}, max_depth={max_depth}, sitemap_seeds={len(sitemap_seeds or [])}"
        )

        # Create content filter for fit markdown generation - optimized for clean content
        # Use request-specific settings or fall back to global config
        min_word_threshold = (
            request.min_word_threshold
            if request.min_word_threshold is not None
            else getattr(settings, "crawl_min_word_threshold", 20)
        )

        # Create a more lenient content filter (unused in this strategy; keep fallback above)
        # Intentionally omitted to avoid unused-variable warnings

        # Use the reusable markdown generator from __init__
        markdown_generator = self.markdown_generator
        if markdown_generator:
            self.logger.debug(
                f"Using reusable markdown generator: {type(markdown_generator)}"
            )
        else:
            # CRITICAL FIX: Create fallback generator to prevent hash placeholders
            self.logger.warning(
                "No markdown generator available - creating fallback to prevent hash placeholders!"
            )
            try:
                from crawl4ai import DefaultMarkdownGenerator  # type: ignore
                from crawl4ai.content_filter_strategy import (
                    PruningContentFilter,  # type: ignore
                )

                # Create emergency fallback generator with conservative settings (match orchestrator config)
                fallback_content_filter = PruningContentFilter(
                    threshold=getattr(
                        settings, "crawl_pruning_threshold", 0.25
                    ),  # Conservative threshold
                    threshold_type="dynamic",  # Dynamic scoring like working single-page
                    min_word_threshold=getattr(
                        settings, "crawl_min_word_threshold", 3
                    ),  # Keep short text blocks
                )
                markdown_generator = DefaultMarkdownGenerator(
                    content_filter=fallback_content_filter
                    # Remove content_source="cleaned_html" - might be causing hash placeholders
                )
                self.logger.info(
                    "Created emergency fallback markdown generator for batch processing"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to create fallback markdown generator: {e} - hash placeholders expected!"
                )

        # High-performance scraping strategy (20x faster parsing)
        scraping_strategy = None
        if getattr(settings, "use_lxml_strategy", True):
            try:
                from crawl4ai.content_scraping_strategy import (
                    LXMLWebScrapingStrategy,  # type: ignore
                )

                # Configure LXML strategy for optimal performance
                scraping_strategy = LXMLWebScrapingStrategy(logger=self.logger)
                self.logger.debug(
                    "Using LXMLWebScrapingStrategy for high-performance parsing"
                )
            except ImportError:
                self.logger.warning(
                    "LXMLWebScrapingStrategy not available, using default"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to configure LXMLWebScrapingStrategy: {e}, using default"
                )

        # Prepare CSS selector filtering parameters
        # Content selector to focus on main content area
        content_selector = (
            request.content_selector
            if request.content_selector is not None
            else getattr(settings, "crawl_content_selector", None)
        )

        # If no content selector specified, optionally use semantic HTML5 selectors
        # This avoids site-specific selectors while targeting main content
        if content_selector is None and getattr(
            settings, "crawl_use_semantic_default_selector", False
        ):
            content_selector = (
                "main, article, .content, [role='main'], .docs-content, .markdown-body"
            )
            self.logger.info(
                "Using semantic HTML5 content selectors for main content detection"
            )

        # Excluded selectors for UI noise removal (join them into a single string)
        excluded_selectors = (
            request.excluded_selectors
            if request.excluded_selectors is not None
            else getattr(settings, "crawl_excluded_selectors", [])
        )
        excluded_selector_string = (
            ",".join(excluded_selectors) if excluded_selectors else None
        )

        # Consistent streaming configuration with CacheMode.BYPASS
        # Always use streaming for better performance with proper cache bypass
        stream_enabled = True
        # Build minimal configuration that actually works with BFS deep crawling
        # Complex filtering parameters interfere with BFS link discovery and following
        if deep_strategy is not None:
            # Minimal BFS-compatible configuration (matches working basic test)
            config_params = {
                "deep_crawl_strategy": deep_strategy,
                "stream": stream_enabled,  # Enable streaming when deep crawl is used
                "cache_mode": cache_mode,
                "page_timeout": page_timeout,
                "semaphore_count": getattr(
                    settings, "crawl_concurrency", 8
                ),  # Optimal for 24-thread CPU
                "verbose": False,  # Disable verbose output for MCP compatibility
                "check_robots_txt": False,  # per user preference
                "markdown_generator": markdown_generator,  # CRITICAL: Add markdown_generator to prevent hash placeholders
                # Remove parameters that interfere with BFS:
                # - excluded_tags: can filter out navigation links
                # - exclude_external_links: prevents following discovered links
                # - word_count_threshold: can filter out pages with links
                # - content selectors: can miss links outside selected areas
            }

            self.logger.info(
                f"BFS Deep crawl config markdown_generator: {config_params.get('markdown_generator', 'NOT_SET')}"
            )
        else:
            # Full configuration for single-page scraping (when deep_strategy is None)
            config_params = {
                "deep_crawl_strategy": deep_strategy,
                "scraping_strategy": scraping_strategy,  # High-performance LXML strategy
                "stream": stream_enabled,
                "cache_mode": cache_mode,
                "page_timeout": page_timeout,
                "semaphore_count": getattr(
                    settings, "crawl_concurrency", 8
                ),  # Optimal for 24-thread CPU
                "remove_overlay_elements": getattr(
                    settings, "crawl_remove_overlays", True
                ),
                "word_count_threshold": max(
                    getattr(settings, "crawl_min_words", 50), min_word_threshold
                ),
                "check_robots_txt": False,
                "verbose": False,
                "excluded_tags": (
                    request.excluded_tags
                    if request.excluded_tags is not None
                    else getattr(
                        settings,
                        "crawl_excluded_tags",
                        ["nav", "footer", "header", "aside", "script", "style"],
                    )
                ),
                "exclude_external_links": True,
                "markdown_generator": markdown_generator,
                "process_iframes": False,
            }

        # Add CSS selector parameters only for single-page scraping
        # CSS selectors can interfere with BFS link discovery by limiting content areas
        if deep_strategy is None:
            optional_params = {
                "css_selector": content_selector,
                "excluded_selector": excluded_selector_string,
            }
            filtered = {k: v for k, v in optional_params.items() if v}
            config_params.update(filtered)
            for k, v in filtered.items():
                msg = (
                    "content selector" if k == "css_selector" else "excluded selectors"
                )
                self.logger.info("Using %s: %s", msg, str(v)[:100])
        else:
            self.logger.info(
                "Skipping CSS selectors for BFS deep crawling to prevent link filtering"
            )

        try:
            run_config = CrawlerRunConfig(**config_params)
        except TypeError as e:
            self.logger.warning(
                "Retrying run config without optional CSS params: %s", e
            )
            for key in ("css_selector", "excluded_selector", "scraping_strategy"):
                config_params.pop(key, None)
            run_config = CrawlerRunConfig(**config_params)

        # Optional: memory thresholds to align with our MemoryManager
        if hasattr(settings, "crawl_memory_threshold_percent"):
            with contextlib.suppress(Exception):
                run_config.memory_threshold_percent = (
                    settings.crawl_memory_threshold_percent
                )  # type: ignore[attr-defined]
        if hasattr(settings, "crawl_memory_check_interval"):
            with contextlib.suppress(Exception):
                run_config.check_interval = settings.crawl_memory_check_interval  # type: ignore[attr-defined]

        # Optional: wait_for selector
        if getattr(request, "wait_for", None):
            run_config.wait_for = request.wait_for  # type: ignore[attr-defined]

        # Extraction strategy (only for single-page scraping)
        # Complex extraction strategies can interfere with BFS link discovery
        if deep_strategy is None:
            extraction_strategy = getattr(request, "extraction_strategy", None)
            if extraction_strategy == "llm":
                with contextlib.suppress(Exception):
                    run_config.extraction_strategy = LLMExtractionStrategy(  # type: ignore[attr-defined]
                        provider="openai",
                        api_token="",
                        instruction="Extract main content and key information from the page",
                    )
            elif extraction_strategy == "cosine":
                with contextlib.suppress(Exception):
                    run_config.extraction_strategy = CosineStrategy(  # type: ignore[attr-defined]
                        semantic_filter="main content, articles, blog posts",
                        word_count_threshold=getattr(settings, "crawl_min_words", 50),
                    )
            # When extraction_strategy is None, crawl4ai will use default content processing
            # with our PruningContentFilter and markdown generator for clean extraction
        else:
            self.logger.info(
                "Skipping extraction strategies for BFS deep crawling to prevent content filtering"
            )

        # Chunking strategy (best-effort; only if available)
        if getattr(request, "chunking_strategy", None):
            try:
                from crawl4ai.chunking_strategy import (  # type: ignore[import-untyped]
                    FixedLengthWordChunking,
                    OverlappingWindowChunking,
                    RegexChunking,
                    SlidingWindowChunking,
                )

                chunking_map = {
                    "overlapping_window": OverlappingWindowChunking,
                    "sliding_window": SlidingWindowChunking,
                    "fixed_length_word": FixedLengthWordChunking,
                    "regex": RegexChunking,
                }
                chunker_class = chunking_map.get(request.chunking_strategy or "")
                if chunker_class:
                    chunking_options = request.chunking_options or {}
                    run_config.chunking_strategy = chunker_class(**chunking_options)  # type: ignore[attr-defined]
            except Exception:
                pass

        return run_config

    def _build_deep_crawl_strategy(
        self, request: CrawlRequest, sitemap_seeds: list[str]
    ) -> BFSDeepCrawlStrategy | None:
        """Create an enhanced BFS deep crawl strategy with advanced filtering and prioritization."""

        # Get crawl limits from request or settings defaults
        max_depth = (
            request.max_depth
            if request.max_depth is not None
            else getattr(settings, "crawl_max_depth", 3)
        )
        max_pages = (
            request.max_pages
            if request.max_pages is not None
            else getattr(settings, "crawl_max_pages", 100)
        )

        self.logger.info(
            "Creating enhanced BFS deep crawl strategy: max_depth=%s, max_pages=%s, sitemap_seeds=%s",
            max_depth,
            max_pages,
            len(sitemap_seeds),
        )

        try:
            # Enhanced BFS strategy with filtering and prioritization
            strategy_params = {
                "max_depth": max_depth,
                "max_pages": max_pages,
                "include_external": False,  # Focus on same-domain content
                # Note: priority_function and filter_function not supported in crawl4ai 0.7.4
            }

            # Note: seed_urls parameter not supported in crawl4ai 0.7.4
            if sitemap_seeds:
                self.logger.debug(
                    f"Found {len(sitemap_seeds)} sitemap seeds (not used in BFSDeepCrawlStrategy)"
                )

            return BFSDeepCrawlStrategy(**strategy_params)  # type: ignore[attr-defined]

        except Exception as e:
            self.logger.warning(
                "Failed to create enhanced BFS strategy, falling back to simple version: %s",
                e,
            )
            # Fallback to simple strategy
            try:
                return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
                    max_depth=max_depth,
                    max_pages=max_pages,
                    include_external=False,
                )
            except Exception as e2:
                self.logger.error(
                    "Failed to create fallback BFS deep crawl strategy: %s", e2
                )
                return None

    def _create_priority_function(self, sitemap_seeds: list[str]):
        """Create a priority function that prefers sitemap URLs and shorter paths."""

        def priority_scorer(url: str, depth: int) -> float:
            """Score URLs for crawling priority (higher = more important)."""
            score = 1.0

            # Prefer sitemap URLs
            if url in sitemap_seeds:
                score += 10.0

            # Prefer shorter depths (closer to root)
            score += max(0, 5 - depth)

            # Prefer documentation-like paths
            doc_indicators = ["doc", "guide", "tutorial", "api", "reference", "help"]
            if any(indicator in url.lower() for indicator in doc_indicators):
                score += 5.0

            # Penalize very long URLs (likely dynamic or less important)
            if len(url) > 100:
                score -= 2.0

            return score

        return priority_scorer

    def _create_url_filter(self, request: CrawlRequest):
        """Create a URL filter function based on request parameters and settings."""
        exclude_patterns = getattr(settings, "crawl_exclude_url_patterns", [])

        def url_filter(url: str) -> bool:
            """Return True if URL should be crawled, False to skip."""
            # Check exclude patterns from settings
            for pattern in exclude_patterns:
                try:
                    if re.search(pattern, url):
                        return False
                except re.error:
                    continue  # Skip invalid regex patterns

            # Additional filtering based on file extensions
            excluded_exts = [
                ".pdf",
                ".jpg",
                ".png",
                ".gif",
                ".mp4",
                ".zip",
                ".exe",
            ]
            return not any(url.lower().endswith(ext) for ext in excluded_exts)

        return url_filter

    async def _discover_sitemap_seeds(self, start_url: str, limit: int) -> list[str]:
        """Fetch robots.txt and sitemap.xml to build seed URLs for prioritization.
        Returns a bounded list of same-domain URLs.
        """
        try:
            parsed = urlparse(start_url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            robots_url = urljoin(base, "/robots.txt")
            sitemap_urls = await self._extract_sitemaps_from_robots(robots_url)
            if not sitemap_urls:
                # fallback to conventional path
                sitemap_urls = [urljoin(base, "/sitemap.xml")]

            seeds: list[str] = []
            for sm in sitemap_urls:
                urls = await self._parse_sitemap(sm, base, remaining=limit - len(seeds))
                seeds.extend(urls)
                if len(seeds) >= limit:
                    break

            # Dedup same-domain
            seen = set()
            same_domain_seeds = []
            for u in seeds:
                try:
                    if urlparse(u).netloc == parsed.netloc and u not in seen:
                        seen.add(u)
                        same_domain_seeds.append(u)
                except Exception:
                    continue
            return same_domain_seeds[:limit]
        except Exception:
            return []

    async def _extract_sitemaps_from_robots(self, robots_url: str) -> list[str]:
        try:
            text = await self._fetch_text(robots_url, timeout=10)
            if not text:
                return []
            sitemaps: list[str] = []
            for line in text.splitlines():
                line = self._safe_strip(line)
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("sitemap:"):
                    sitemaps.append(self._safe_strip(line.split(":", 1)[1]))
            return sitemaps
        except Exception:
            return []

    async def _parse_sitemap(
        self, sitemap_url: str, base: str, remaining: int
    ) -> list[str]:
        """Parse a sitemap or sitemap index and return up to `remaining` URLs."""
        try:
            import xml.etree.ElementTree as ET

            xml_text = await self._fetch_text(sitemap_url, timeout=15)
            if not xml_text:
                return []
            urls: list[str] = []
            try:
                root = ET.fromstring(xml_text)
            except Exception:
                return []

            tag = root.tag.lower()

            def ns_strip(t: str) -> str:
                return t.split("}", 1)[-1] if "}" in t else t

            tag = ns_strip(tag)
            if tag == "sitemapindex":
                for sm in root.findall(".//{*}sitemap/{*}loc"):
                    loc_text = self._safe_strip(sm.text or "")
                    if not loc_text:
                        continue
                    if len(urls) >= remaining:
                        break
                    urls.extend(
                        await self._parse_sitemap(loc_text, base, remaining - len(urls))
                    )
                    if len(urls) >= remaining:
                        break
            elif tag == "urlset":
                for loc in root.findall(".//{*}url/{*}loc"):
                    loc_text = self._safe_strip(loc.text or "")
                    if not loc_text:
                        continue
                    urls.append(loc_text)
                    if len(urls) >= remaining:
                        break
            # Normalize to absolute
            abs_urls: list[str] = []
            for u in urls[:remaining]:
                try:
                    abs_urls.append(urljoin(base, u))
                except Exception:
                    continue
            return abs_urls[:remaining]
        except Exception:
            return []

    async def _fetch_text(self, url: str, timeout: int = 10) -> str:
        """Lightweight async fetch (Playwright via the existing crawler session is not used here to avoid side effects).
        Uses aiohttp if available, else returns empty string.
        """
        try:
            import aiohttp  # type: ignore
        except Exception:
            return ""

        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with (
                aiohttp.ClientSession() as session,
                session.get(url, timeout=timeout_obj) as resp,
            ):
                if resp.status != 200:
                    return ""
                text_content = await resp.text()
                return str(text_content)
        except Exception:
            return ""

    async def _crawl_using_deep_strategy(
        self, browser: Any, start_url: str, run_config: Any, max_pages: int
    ) -> tuple[list[Any], list[str]]:
        """Crawl using BFSDeepCrawlStrategy with async generator."""
        successful_results = []
        errors = []

        with suppress_stdout():
            try:
                # Get result from arun - type depends on config.stream setting
                self.logger.info(
                    "About to call browser.arun with stream=%s", run_config.stream
                )
                crawl_result = await browser.arun(url=start_url, config=run_config)
                self.logger.info("browser.arun completed successfully")

                # Debug: Log the actual type we received
                self.logger.info(
                    "CRAWL DEBUG: crawl_result type = %s, stream=%s, deep_crawl=%s",
                    type(crawl_result).__name__,
                    run_config.stream,
                    run_config.deep_crawl_strategy is not None,
                )

                # Handle both streaming and non-streaming results from BFSDeepCrawlStrategy
                self.logger.info(f"Processing BFS results - type: {type(crawl_result)}")
                self.logger.info(
                    f"DEBUG: crawl_result has __aiter__: {hasattr(crawl_result, '__aiter__')}"
                )

                generator_count = 0
                results_to_process = []

                try:
                    # Check if we have streaming results (async generator)
                    if hasattr(crawl_result, "__aiter__"):
                        self.logger.info(
                            "Processing streaming results (async generator)"
                        )
                        async for result in crawl_result:
                            results_to_process.append(result)
                            generator_count += 1
                    else:
                        # Non-streaming mode - handle direct results
                        self.logger.info("Processing non-streaming results")
                        if isinstance(crawl_result, list):
                            results_to_process = crawl_result
                        else:
                            results_to_process = [crawl_result]
                        generator_count = len(results_to_process)

                    # Process all collected results
                    for idx, result in enumerate(results_to_process, 1):
                        self.logger.info(
                            f"Processing result #{idx}/{len(results_to_process)}: {result.url if hasattr(result, 'url') else type(result).__name__}"
                        )

                        # Pre-check for unexpected types (defensive programming)
                        if isinstance(result, int):
                            self.logger.warning(
                                "Received integer %d instead of CrawlResult in streaming mode, skipping",
                                result,
                            )
                            continue

                        # Ensure result is a CrawlResult object
                        if not hasattr(result, "success"):
                            self.logger.warning(
                                "Received unexpected type %s in streaming mode, skipping",
                                type(result).__name__,
                            )
                            continue

                        if result.success:
                            try:
                                successful_results.append(result)
                            except AttributeError as e:
                                if (
                                    "'int' object has no attribute 'raw_markdown'"
                                    in str(e)
                                ):
                                    self.logger.warning(
                                        "Caught integer markdown hash issue for %s, skipping result",
                                        result.url,
                                    )
                                    continue
                                else:
                                    raise
                        else:
                            errors.append(
                                f"Failed to crawl {result.url}: {result.error_message}"
                            )
                        if len(successful_results) >= max_pages:
                            self.logger.info(
                                f"Breaking from AsyncGenerator loop: reached max_pages ({max_pages})"
                            )
                            break

                    self.logger.info(
                        f"Deep crawl processing completed: processed {len(results_to_process)} results, {len(successful_results)} successful"
                    )
                except Exception as async_error:
                    self.logger.error(
                        f"Async iteration failed: {async_error}", exc_info=True
                    )
                    errors.append(f"Async iteration error: {async_error}")

            except Exception as e:
                self.logger.error(f"Deep crawl strategy failed: {e}", exc_info=True)
                errors.append(str(e))

        return successful_results, errors

    async def _crawl_using_arun_many(
        self,
        browser: Any,
        sitemap_urls: list[str],
        run_config: Any,
        request: Any,
        progress_callback: Any,
    ) -> tuple[list[Any], list[str]]:
        """Crawl using arun_many() with discovered sitemap URLs."""
        try:
            from crawl4ai import MemoryAdaptiveDispatcher  # type: ignore
        except ImportError as e:
            logger.error(
                "MemoryAdaptiveDispatcher not available from crawl4ai: %s. "
                "Falling back to sequential crawling.",
                e,
            )
            MemoryAdaptiveDispatcher = None
        except Exception as e:
            logger.error(
                "Unexpected error importing MemoryAdaptiveDispatcher: %s. "
                "Falling back to sequential crawling.",
                e,
            )
            MemoryAdaptiveDispatcher = None

        successful_results: list[Any] = []
        errors: list[str] = []
        max_pages = request.max_pages or len(sitemap_urls)
        # Optimal concurrency for 24-thread i7-13700K: 8 concurrent sessions
        max_concurrent = getattr(settings, "crawl_concurrency", 8)

        # Limit sitemap URLs to max_pages
        urls_to_crawl = sitemap_urls[:max_pages]

        # Check if MemoryAdaptiveDispatcher is available
        if MemoryAdaptiveDispatcher is None:
            logger.warning(
                "MemoryAdaptiveDispatcher unavailable, falling back to sequential crawling"
            )
            # Fallback to sequential crawling
            for url in urls_to_crawl:
                try:
                    result = await browser.arun(url=url, config=run_config)
                    successful_results.append(result)
                except Exception as e:
                    error_msg = f"Error crawling {url}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            return successful_results, errors

        self.logger.info(
            f"Creating MemoryAdaptiveDispatcher with max_session_permit={max_concurrent}"
        )

        # Create optimized dispatcher for memory-adaptive concurrency
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=getattr(
                settings, "crawl_memory_threshold", 70.0
            ),  # Optimized for 32GB+ RAM
            check_interval=0.5,  # Faster checks for optimal responsiveness with i7-13700k
            max_session_permit=max_concurrent,
        )

        # Remove deep_crawl_strategy to avoid recursion and optimize batch configuration
        batch_config = (
            run_config.clone() if hasattr(run_config, "clone") else run_config
        )
        if hasattr(batch_config, "deep_crawl_strategy"):
            batch_config.deep_crawl_strategy = None

        # Enable streaming for optimal performance with proper content extraction
        batch_config.stream = (
            True  # CRITICAL: Enable streaming for concurrent processing
        )

        # Ensure proper page loading time for content extraction
        if hasattr(batch_config, "delay_before_return_html"):
            batch_config.delay_before_return_html = (
                1.0  # Balanced delay for performance vs completeness
            )

        # Connection pool optimization for concurrent sessions
        if hasattr(batch_config, "network_config"):
            batch_config.network_config = {
                "connection_pool_size": 32,  # Match config.py qdrant_connection_pool_size
                "keep_alive": True,
                "http2": True,  # Enable HTTP/2 for multiplexing
                "timeout": 30.0,  # Reasonable timeout for batch processing
            }

        # Virtual scrolling optimizations for batch processing
        if hasattr(batch_config, "virtual_scroll_config") or hasattr(
            batch_config, "virtual_scroll"
        ):
            # Enable virtual scrolling with optimized settings for concurrent processing
            scroll_count = getattr(settings, "crawl_scroll_count", 20)
            batch_size = getattr(settings, "crawl_virtual_scroll_batch_size", 5)

            if hasattr(batch_config, "virtual_scroll_config"):
                batch_config.virtual_scroll_config = {
                    "scroll_count": scroll_count,
                    "batch_size": batch_size,  # Process in smaller batches for better concurrency
                    "scroll_pause": 0.3,  # Faster scrolling for batch processing
                    "viewport_height": 800,  # Consistent with browser config
                }
            elif hasattr(batch_config, "virtual_scroll_count"):
                batch_config.virtual_scroll_count = scroll_count

        # Use the reusable markdown generator for arun_many
        if (
            not hasattr(batch_config, "markdown_generator")
            or batch_config.markdown_generator is None
        ):
            batch_config.markdown_generator = self.markdown_generator
            if self.markdown_generator:
                self.logger.debug(
                    f"Using reusable markdown_generator for arun_many: {type(self.markdown_generator)}"
                )
            else:
                # CRITICAL FIX: Create fallback generator for arun_many to prevent hash placeholders
                self.logger.warning(
                    "No markdown generator available for arun_many - creating fallback to prevent hash placeholders!"
                )
                try:
                    from crawl4ai import DefaultMarkdownGenerator  # type: ignore
                    from crawl4ai.content_filter_strategy import (
                        PruningContentFilter,  # type: ignore
                    )

                    # Create emergency fallback generator with conservative settings for arun_many (match orchestrator)
                    fallback_content_filter = PruningContentFilter(
                        threshold=getattr(
                            settings, "crawl_pruning_threshold", 0.25
                        ),  # Conservative threshold
                        threshold_type="dynamic",  # Dynamic scoring like working single-page
                        min_word_threshold=getattr(
                            settings, "crawl_min_word_threshold", 3
                        ),  # Keep short text blocks
                    )
                    batch_config.markdown_generator = DefaultMarkdownGenerator(
                        content_filter=fallback_content_filter
                        # Remove content_source="cleaned_html" - might be causing hash placeholders
                    )
                    self.logger.info(
                        "Created emergency fallback markdown generator for arun_many batch processing"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to create fallback markdown generator for arun_many: {e} - hash placeholders expected!"
                    )
        else:
            self.logger.debug(
                f"arun_many batch_config already has markdown_generator: {type(batch_config.markdown_generator)}"
            )

        self.logger.info(f"Starting arun_many with {len(urls_to_crawl)} URLs")

        with suppress_stdout():
            try:
                # Use arun_many for concurrent crawling
                results_generator = await browser.arun_many(
                    urls=urls_to_crawl, config=batch_config, dispatcher=dispatcher
                )

                processed_count = 0
                async for result in results_generator:
                    processed_count += 1
                    self.logger.info(
                        f"arun_many result #{processed_count}: {result.url if hasattr(result, 'url') else type(result).__name__}"
                    )

                    if hasattr(result, "success") and result.success:
                        try:
                            successful_results.append(result)

                            if progress_callback:
                                progress_callback(
                                    len(successful_results),
                                    max_pages,
                                    f"Crawled {result.url}",
                                )

                        except Exception as e:
                            self.logger.warning(
                                "Failed to process result for %s: %s",
                                getattr(result, "url", "unknown"),
                                e,
                            )
                            errors.append(str(e))

                    if len(successful_results) >= max_pages:
                        self.logger.info(f"Reached max_pages limit ({max_pages})")
                        break

                self.logger.info(
                    f"arun_many completed: {processed_count} processed, {len(successful_results)} successful"
                )

            except Exception as e:
                self.logger.error("arun_many approach failed: %s", e, exc_info=True)
                # Fallback to single URL if arun_many fails
                if urls_to_crawl:
                    self.logger.info("Falling back to single URL crawl")
                    single_result = await browser.arun(
                        url=urls_to_crawl[0], config=batch_config
                    )

                    if hasattr(single_result, "success") and single_result.success:
                        successful_results.append(single_result)
                    else:
                        errors.append(
                            f"Failed to crawl {getattr(single_result, 'url', urls_to_crawl[0])}"
                        )

        return successful_results, errors

    async def _crawl_using_concurrent_arun(
        self,
        browser: Any,
        urls: list[str],
        run_config: Any,
        request: CrawlRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> tuple[list[Any], list[str]]:
        """
        Concurrent crawling using individual arun() calls with asyncio.gather().
        This bypasses the arun_many() hash placeholder bug while maintaining performance.
        """
        import asyncio
        from typing import Any as TypingAny

        successful_results = []
        errors = []
        max_concurrent = getattr(settings, "crawl_concurrency", 8)

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        self.logger.info(
            f"Starting concurrent arun with {len(urls)} URLs, max concurrency: {max_concurrent}"
        )

        # Remove stream setting for individual arun calls to ensure proper content extraction
        single_config = (
            run_config.clone() if hasattr(run_config, "clone") else run_config
        )
        if hasattr(single_config, "stream"):
            single_config.stream = False  # Disable streaming for individual calls
        if hasattr(single_config, "deep_crawl_strategy"):
            single_config.deep_crawl_strategy = (
                None  # Remove deep crawl for individual calls
            )

        async def crawl_single_url(
            url: str, index: int
        ) -> tuple[TypingAny | None, str | None]:
            """Crawl a single URL with semaphore limiting."""
            async with semaphore:
                try:
                    self.logger.debug(f"Crawling URL {index + 1}/{len(urls)}: {url}")
                    result = await browser.arun(url=url, config=single_config)

                    if result.success:
                        if progress_callback:
                            progress_callback(
                                index + 1,
                                len(urls),
                                f"Crawled {url}",
                            )
                        return result, None
                    else:
                        error_msg = f"Failed to crawl {url}: {result.error_message}"
                        self.logger.warning(error_msg)
                        return None, error_msg

                except Exception as e:
                    error_msg = f"Exception crawling {url}: {e}"
                    self.logger.error(error_msg)
                    return None, error_msg

        # Create tasks for all URLs
        tasks = [crawl_single_url(url, i) for i, url in enumerate(urls)]

        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"Task exception for {urls[i]}: {result}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                else:
                    crawl_result, error = result
                    if crawl_result:
                        successful_results.append(crawl_result)
                    if error:
                        errors.append(error)

        except Exception as e:
            self.logger.error(f"Concurrent crawling failed: {e}")
            errors.append(f"Concurrent crawling error: {e}")

        self.logger.info(
            f"Concurrent arun completed: {len(successful_results)} successful, {len(errors)} errors"
        )

        return successful_results, errors

    async def pre_execute_setup(self) -> None:
        """Setup before crawling begins."""
        await super().pre_execute_setup()
        # No browser session cleanup needed - each crawl uses fresh browser

    async def _crawl_using_orchestrator_approach(
        self,
        browser: Any,
        urls: list[str],
        request: CrawlRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> tuple[list[Any], list[str]]:
        """
        CRITICAL FIX: Use the same approach as orchestrator.scrape_single_page to avoid hash placeholder bug.
        This method replicates the working single-page logic for each URL sequentially.
        """
        successful_results = []
        errors = []

        self.logger.info(
            f"Starting orchestrator-style sequential crawling of {len(urls)} URLs"
        )

        for i, url in enumerate(urls):
            try:
                self.logger.info(f"Crawling URL {i + 1}/{len(urls)}: {url}")

                # Use the exact same configuration as the working orchestrator approach
                from ..types.crawl4ai_types import (
                    DefaultMarkdownGeneratorImpl as DefaultMarkdownGenerator,
                )
                from ..types.crawl4ai_types import (
                    PruningContentFilterImpl as PruningContentFilter,
                )

                # Create content filter exactly like orchestrator does (lines 266-270 in orchestrator.py)
                content_filter = None
                if PruningContentFilter is not None:
                    content_filter = PruningContentFilter(
                        threshold=0.25,  # EXACT same as orchestrator
                        threshold_type="dynamic",  # EXACT same as orchestrator
                        min_word_threshold=3,  # EXACT same as orchestrator
                    )

                # Create markdown generator exactly like orchestrator does (lines 273-277)
                markdown_generator = None
                if DefaultMarkdownGenerator is not None:
                    markdown_generator = DefaultMarkdownGenerator(
                        content_filter=content_filter
                        # No content_source parameter - exactly like orchestrator
                    )

                # Configure crawl parameters exactly like orchestrator does (lines 279-293)
                crawl_kwargs = {
                    "url": url,
                    "bypass_cache": not settings.crawl_enable_caching,
                    "process_iframes": False,
                    "remove_overlay_elements": settings.crawl_remove_overlays,
                    "word_count_threshold": settings.crawl_min_words,
                    # EXACT same excluded tags as orchestrator
                    "excluded_tags": [
                        "script",
                        "style",
                    ],
                    "exclude_external_links": True,
                    "markdown_generator": markdown_generator,  # Use the working generator
                }

                # Crawl the page using browser.arun (same as orchestrator line 331)
                result = await browser.arun(**crawl_kwargs)

                if result.success:
                    # CRITICAL: Apply the same content extraction logic as orchestrator (lines 337-371)
                    # to avoid the broken _extract_markdown method in _to_page_content
                    best_content = ""
                    if result.markdown:
                        try:
                            # Extract markdown content following crawl4ai best practices (same as orchestrator)
                            if (
                                hasattr(result.markdown, "fit_markdown")
                                and result.markdown.fit_markdown
                            ):
                                best_content = result.markdown.fit_markdown.strip()
                            elif (
                                hasattr(result.markdown, "raw_markdown")
                                and result.markdown.raw_markdown
                            ):
                                best_content = result.markdown.raw_markdown.strip()
                            else:
                                best_content = ""
                        except Exception as e:
                            self.logger.debug(
                                f"Failed to extract markdown content for {url}: {e}"
                            )
                            best_content = ""

                    # Basic content validation (same as orchestrator)
                    if best_content and len(best_content.strip()) < 3:
                        self.logger.debug(
                            f"Content too short for {url} (less than 3 chars), clearing"
                        )
                        best_content = ""

                    # Simple fallback if markdown extraction failed (same as orchestrator)
                    if not best_content and hasattr(result, "text") and result.text:
                        best_content = str(result.text).strip()
                        self.logger.debug(f"Using text fallback for {url}")

                    if (
                        not best_content
                        and hasattr(result, "cleaned_text")
                        and result.cleaned_text
                    ):
                        best_content = str(result.cleaned_text).strip()
                        self.logger.debug(f"Using cleaned_text fallback for {url}")

                    # Store the extracted content directly in the result for _to_page_content to use
                    # This bypasses the broken _extract_markdown method
                    result._orchestrator_extracted_content = best_content
                    result._orchestrator_word_count = (
                        len(best_content.split()) if best_content else 0
                    )

                    successful_results.append(result)
                    self.logger.info(
                        f"Successfully crawled {url} with {result._orchestrator_word_count} words"
                    )

                    if progress_callback:
                        progress_callback(
                            i + 1,
                            len(urls),
                            f"Crawled {url}",
                        )
                else:
                    error_msg = f"Failed to crawl {url}: {result.error_message}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)

            except Exception as e:
                error_msg = f"Exception crawling {url}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)

        self.logger.info(
            f"Orchestrator-style crawling completed: {len(successful_results)} successful, {len(errors)} errors"
        )
        return successful_results, errors
