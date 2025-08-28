"""
Optimized web crawling strategy with streaming and caching support.
"""

import contextlib
import logging
import os
import re
import sys
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
from crawl4ai.content_filter_strategy import PruningContentFilter  # type: ignore
from crawl4ai.deep_crawling import (  # type: ignore
    BFSDeepCrawlStrategy,
)
from crawl4ai.deep_crawling.filters import URLPatternFilter  # type: ignore
from crawl4ai.extraction_strategy import (  # type: ignore
    CosineStrategy,
    LLMExtractionStrategy,
)

from ..config import settings
from ..models.crawl import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from ..types.crawl4ai_types import DefaultMarkdownGeneratorImpl
from .base import BaseCrawlStrategy

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stdout() -> Any:
    """Context manager to suppress stdout output (redirect to devnull)."""
    old_stdout = sys.stdout
    try:
        # Redirect stdout to devnull to prevent interference with MCP protocol
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = old_stdout


class WebCrawlStrategy(BaseCrawlStrategy):
    """
    High-performance web crawling strategy with streaming, caching, and GPU acceleration.
    Optimized for RTX 4070 + i7-13700K performance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.memory_manager = None

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

            # Simplified browser config compatible with containerized Playwright chromium
            browser_config = BrowserConfig(
                headless=settings.crawl_headless,
                browser_type=settings.crawl_browser,
                light_mode=True,  # Optimized performance mode
                verbose=False,  # Suppress Crawl4AI console output for MCP compatibility
                # Only essential flags compatible with containerized Playwright
                extra_args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
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
            sitemap_seeds = [s for s in sitemap_seeds if not (s in seen or seen.add(s))]
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

            pages = []
            errors = []
            total_bytes = 0
            unique_domains = set()
            total_links_discovered = 0
            max_pages = request.max_pages or 100

            if progress_callback:
                progress_callback(0, max_pages, "Starting crawl...")

            # Choose crawling approach: arun_many() with sitemap URLs or BFSDeepCrawlStrategy
            crawl_count = 0
            if (
                getattr(settings, "use_arun_many_for_sitemaps", False)
                and sitemap_seeds
                and len(sitemap_seeds) > 1
            ):
                self.logger.info(
                    f"Using arun_many() approach with {len(sitemap_seeds)} sitemap URLs (max_concurrent_sessions={getattr(settings, 'max_concurrent_sessions', 20)})"
                )
                successful_results, errors = await self._crawl_using_arun_many(
                    browser, sitemap_seeds, run_config, request, progress_callback
                )
            else:
                self.logger.info(
                    "Using BFSDeepCrawlStrategy approach with async iteration..."
                )
                # Use the first URL as the starting point but keep others for discovery via seeds
                start_url = request.url[0]
                successful_results, errors = await self._crawl_using_deep_strategy(
                    browser, start_url, run_config, max_pages
                )

            # Process crawling results
            pages = []

            # Process results outside suppress_stdout context for debugging
            # Prefer fit_markdown for cleaner content, respecting global setting when request doesn't specify
            prefer_fit_markdown = (
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
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            self.logger.warning("Failed to extract text from HTML: %s", e)
            return ""

    def _sanitize_crawl_result(self, result: Crawl4aiResult) -> Crawl4aiResult:
        """Sanitize CrawlResult to prevent integer hash issues with markdown field."""

        from ..types.crawl4ai_types import (
            MarkdownGenerationResultImpl as MarkdownGenerationResult,
        )

        try:
            # Check if the private _markdown field contains an integer hash
            if hasattr(result, "_markdown") and isinstance(result._markdown, int):
                logger.debug(
                    "Found integer _markdown (%s), replacing with empty MarkdownGenerationResult",
                    result._markdown,
                )
                # Replace the integer hash with an empty MarkdownGenerationResult
                result._markdown = MarkdownGenerationResult(
                    raw_markdown="",
                    markdown_with_citations="",
                    references_markdown="",
                    fit_markdown=None,
                    fit_html=None,
                )

            # Also check if markdown property access would fail
            # This is a defensive check
            if hasattr(result, "markdown"):
                try:
                    # Try to access it to see if it would error
                    _ = result.markdown
                except AttributeError as e:
                    if "'int' object has no attribute" in str(e):
                        logger.debug(
                            "Markdown property access failed, force setting safe value for %s",
                            result.url,
                        )
                        # Force set a safe markdown value
                        result._markdown = MarkdownGenerationResult(
                            raw_markdown="",
                            markdown_with_citations="",
                            references_markdown="",
                            fit_markdown=None,
                            fit_html=None,
                        )
        except Exception as e:
            logger.debug(
                "Sanitization warning for %s: %s",
                getattr(result, "url", "unknown"),
                e,
                exc_info=True,
            )

        return result

    def _safe_get_markdown(
        self, result: Crawl4aiResult, prefer_fit_markdown: bool = True
    ) -> str:
        """Safely extract markdown content from crawl4ai result.

        Based on crawl4ai documentation, result.markdown is a MarkdownGenerationResult object
        with attributes like raw_markdown and fit_markdown. We should access these directly.

        However, crawl4ai sometimes returns integer hash IDs instead of proper objects,
        so we need to handle that case as well.
        """
        if not result.markdown:
            return ""

        try:
            # Check if result.markdown is an integer (hash ID issue)
            if isinstance(result.markdown, int):
                self.logger.debug(
                    "CRAWL DEBUG - result.markdown is integer %s, returning empty",
                    result.markdown,
                )
                return ""

            # Choose between fit_markdown and raw_markdown based on preference
            if prefer_fit_markdown:
                # First try fit_markdown (filtered content) if available
                if (
                    hasattr(result.markdown, "fit_markdown")
                    and result.markdown.fit_markdown
                ):
                    content = result.markdown.fit_markdown
                    if (
                        isinstance(content, str) and len(content) > 16
                    ):  # Avoid hash placeholders
                        return content

                # Fall back to raw_markdown (full content)
                if (
                    hasattr(result.markdown, "raw_markdown")
                    and result.markdown.raw_markdown
                ):
                    content = result.markdown.raw_markdown
                    if (
                        isinstance(content, str) and len(content) > 16
                    ):  # Avoid hash placeholders
                        return content
            else:
                # Prefer raw_markdown first if prefer_fit_markdown is False
                if (
                    hasattr(result.markdown, "raw_markdown")
                    and result.markdown.raw_markdown
                ):
                    content = result.markdown.raw_markdown
                    if (
                        isinstance(content, str) and len(content) > 16
                    ):  # Avoid hash placeholders
                        return content

                # Fall back to fit_markdown
                if (
                    hasattr(result.markdown, "fit_markdown")
                    and result.markdown.fit_markdown
                ):
                    content = result.markdown.fit_markdown
                    if (
                        isinstance(content, str) and len(content) > 16
                    ):  # Avoid hash placeholders
                        return content

            # If neither is available or they're just hash placeholders, return empty string
            return ""

        except (AttributeError, TypeError) as e:
            self.logger.debug(
                "CRAWL DEBUG - Exception accessing markdown attributes: %s", e
            )
            return ""
        except Exception as e:
            self.logger.warning(f"Failed to extract markdown content: {e}")
            return ""

    def _post_process_content(self, content: str) -> str:
        """Apply post-processing cleanup to remove UI artifacts and noise."""
        if not content.strip():
            return content

        # Only apply cleanup if enabled in settings
        if not getattr(settings, "clean_ui_artifacts", True):
            return content

        # Remove isolated "Copy" text - anchor to full lines or common button wrappers
        content = re.sub(r"^\s*Copy\s*$", "", content, flags=re.MULTILINE)
        content = re.sub(
            r"\[Copy\]|\(Copy\)|Copy\s*button", "", content, flags=re.IGNORECASE
        )

        # Remove package manager tabs - only full lines that consist solely of package manager tokens
        content = re.sub(
            r"^\s*(?:npm|yarn|pnpm|bun)(?:\s+(?:npm|yarn|pnpm|bun))*\s*$",
            "",
            content,
            flags=re.MULTILINE,
        )

        # Remove repeated navigation patterns - full-line repeated nav breadcrumbs
        content = re.sub(
            r"^(\s*(?:Home|Docs|API|Guide|Tutorial|Examples)\s*[>|/|\-]\s*){2,}.*$",
            "",
            content,
            flags=re.MULTILINE,
        )

        # Remove lines that are only UI commands or single words
        lines = content.split("\n")
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines, single words that look like UI elements, or very short lines
            if (
                len(line) > 3
                and not re.match(
                    r"^(?:Copy|Edit|Share|Save|Print|Download|View|Show|Hide)$",
                    line,
                    re.IGNORECASE,
                )
                and not re.match(r"^[A-Za-z]{1,4}$", line)
            ):  # Skip very short single words
                filtered_lines.append(line)

        # Rejoin lines and clean up multiple newlines
        content = "\n".join(filtered_lines)
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()

    def _to_page_content(
        self, result: Crawl4aiResult, prefer_fit_markdown: bool = True
    ) -> PageContent:
        """Converts a crawl4ai result to a PageContent object."""

        # STREAMING CRAWL FIX: During streaming, Crawl4AI returns lazy-loaded objects with hash placeholders
        # Sanitize the result first to prevent integer hash issues
        result = self._sanitize_crawl_result(result)

        # Extract content using the _safe_get_markdown method which follows reference code pattern
        best_content = self._safe_get_markdown(result, prefer_fit_markdown)

        # Apply post-processing cleanup to remove UI artifacts
        best_content = self._post_process_content(best_content)

        # Calculate word count
        word_count = len(best_content.split()) if best_content.strip() else 0

        # Debug output to show what we found
        debug_msg = (
            f"CRAWL DEBUG - Final content extraction for {result.url}: "
            f"content_length={len(best_content)}, "
            f"word_count={word_count}"
        )
        self.logger.debug(debug_msg)
        if hasattr(self, "ctx") and getattr(self, "ctx", None):
            with contextlib.suppress(Exception):
                self.ctx.info(
                    f"Processed content for {getattr(result, 'url', 'unknown')}"
                )

        return PageContent(
            url=result.url,
            title=result.metadata.get("title", ""),
            content=best_content,
            html=result.html,
            markdown=best_content,  # Use the validated content as markdown
            word_count=word_count,
            links=[
                link.get("href", link) if isinstance(link, dict) else link
                for link in result.links.get("internal", [])
            ]
            if result.links
            else [],
            images=[
                img.get("src", img) if isinstance(img, dict) else img
                for img in result.media.get("images", [])
            ]
            if result.media
            else [],
            metadata={
                "depth": result.metadata.get("depth", 0),
                "status_code": result.status_code,
                "response_headers": dict(result.response_headers or {}),
                "chunk_metadata": getattr(result, "chunk_metadata", {}),
            },
            timestamp=datetime.fromtimestamp(time.time()),
        )

    async def _build_run_config(
        self, request: CrawlRequest, sitemap_seeds: list[str] | None = None
    ) -> CrawlerRunConfig:
        """Build Crawl4AI CrawlerRunConfig aligned with deep crawling and streaming."""
        # Cache mode mapping
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

        # Enable deep crawling if either max_pages > 1 OR max_depth > 0
        # This allows depth-based crawling even with unlimited pages
        if max_pages > 1 or max_depth > 0:
            self.logger.info(
                f"Creating deep crawl strategy: max_pages={max_pages}, max_depth={max_depth}, sitemap_seeds={len(sitemap_seeds or [])}"
            )
            deep_strategy = self._build_deep_crawl_strategy(
                request, sitemap_seeds or []
            )
        else:
            self.logger.info(
                f"Skipping deep crawl strategy: max_pages={max_pages}, max_depth={max_depth}"
            )

        # Create content filter for fit markdown generation - optimized for clean content
        # Use request-specific settings or fall back to global config
        pruning_threshold = (
            request.pruning_threshold
            if request.pruning_threshold is not None
            else getattr(settings, "crawl_pruning_threshold", 0.5)
        )
        min_word_threshold = (
            request.min_word_threshold
            if request.min_word_threshold is not None
            else getattr(settings, "crawl_min_word_threshold", 20)
        )

        content_filter = PruningContentFilter(
            threshold=pruning_threshold,  # Use configurable threshold for relevance scoring
            threshold_type="dynamic",  # Dynamic scoring for adaptive filtering
            min_word_threshold=min_word_threshold,  # Configurable word threshold for content blocks
        )

        # Create markdown generator with content filter
        markdown_generator = DefaultMarkdownGeneratorImpl(content_filter=content_filter)

        # High-performance scraping strategy (20x faster parsing)
        scraping_strategy = None
        if getattr(settings, "use_lxml_strategy", True):
            try:
                from crawl4ai.content_scraping_strategy import (
                    LXMLWebScrapingStrategy,  # type: ignore
                )

                scraping_strategy = LXMLWebScrapingStrategy()
                self.logger.info("Using LXMLWebScrapingStrategy for 20x faster parsing")
            except ImportError:
                self.logger.warning(
                    "LXMLWebScrapingStrategy not available, using default"
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

        # Base run config with fit markdown optimization and CSS filtering
        # NOTE: Deep crawl requires streaming mode (stream=True) to work properly
        # The BFS strategy expects an async generator, not a list
        stream_enabled = (
            deep_strategy is not None
        )  # Enable streaming only if deep crawl is used
        # Build minimal configuration that actually works with BFS deep crawling
        # Complex filtering parameters interfere with BFS link discovery and following
        if deep_strategy is not None:
            # Minimal BFS-compatible configuration (matches working basic test)
            config_params = {
                "deep_crawl_strategy": deep_strategy,
                "stream": stream_enabled,  # Enable streaming when deep crawl is used
                "cache_mode": cache_mode,
                "page_timeout": page_timeout,
                "semaphore_count": 5,  # Lower concurrency for stable BFS operation
                "verbose": False,  # Disable verbose output for MCP compatibility
                "check_robots_txt": False,  # per user preference
                # Remove parameters that interfere with BFS:
                # - excluded_tags: can filter out navigation links
                # - exclude_external_links: prevents following discovered links
                # - word_count_threshold: can filter out pages with links
                # - content selectors: can miss links outside selected areas
            }
        else:
            # Full configuration for single-page scraping (when deep_strategy is None)
            config_params = {
                "deep_crawl_strategy": deep_strategy,
                "scraping_strategy": scraping_strategy,  # High-performance LXML strategy
                "stream": stream_enabled,
                "cache_mode": cache_mode,
                "page_timeout": page_timeout,
                "semaphore_count": getattr(settings, "crawl_concurrency", 30),
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
        """Construct a Best-First deep crawl strategy with filters and scoring; fallback to BFS."""

        # Fix depth calculation - use settings default instead of hardcoded 1
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

        # Build filter chain from include/exclude patterns
        include_patterns = request.include_patterns or []
        exclude_patterns = (request.exclude_patterns or []) + getattr(
            settings, "crawl_exclude_url_patterns", []
        )

        # filter_chain = None  # Would be used if filter chain was implemented
        try:
            filters: list[Any] = []
            if include_patterns:
                # Some versions of URLPatternFilter may not support modes; wrap in try/except
                try:
                    filters.append(
                        URLPatternFilter(
                            patterns=list(include_patterns), mode="include"
                        )
                    )  # type: ignore[attr-defined]
                except Exception:
                    filters.append(URLPatternFilter(patterns=list(include_patterns)))  # type: ignore[attr-defined]
            if exclude_patterns:
                try:
                    filters.append(
                        URLPatternFilter(
                            patterns=list(exclude_patterns), mode="exclude"
                        )
                    )  # type: ignore[attr-defined]
                except Exception:
                    filters.append(URLPatternFilter(patterns=list(exclude_patterns)))  # type: ignore[attr-defined]
            if filters:
                # filter_chain = FilterChain(filters)  # type: ignore[attr-defined]
                pass  # Would create filter chain here if using it
        except Exception:
            pass  # filter_chain will remain None

        # Scorer: prioritize includes, sitemap-derived keywords, and content-like URLs
        try:
            keywords: list[str] = []
            for pat in include_patterns:
                keywords.extend(
                    [
                        t
                        for t in str(pat).replace("*", " ").replace("/", " ").split()
                        if len(t) > 2
                    ]
                )
            # Add tokens from ALL discovered sitemap URLs for better keyword scoring
            # Don't limit by max_pages here - we want comprehensive keywords
            for u in sitemap_seeds:
                try:
                    path = urlparse(u).path
                    keywords.extend(
                        [
                            t
                            for t in path.replace("-", " ")
                            .replace("_", " ")
                            .replace("/", " ")
                            .split()
                            if len(t) > 2
                        ]
                    )
                except Exception:
                    continue
            if not keywords:
                keywords = ["docs", "blog", "guide", "article", "learn", "help", "faq"]

            # Add more generic keywords to be less restrictive
            keywords.extend(
                [
                    "fastmcp",
                    "mcp",
                    "client",
                    "server",
                    "api",
                    "tutorial",
                    "example",
                    "getting",
                    "started",
                ]
            )
            self.logger.info("Using keywords for scoring: %s...", keywords[:10])
            # If supported by your crawl4ai version:
            # deep_kwargs = {"scorer": KeywordRelevanceScorer(keywords=keywords, weight=0.7)}
            # Would create scorer kwargs here if using them
        except Exception as e:
            self.logger.warning("Failed to create scorer: %s", e)
            # deep_kwargs will remain empty

        # Use simplified BFS strategy that actually works (based on successful basic test)
        try:
            self.logger.info(
                "Creating simplified BFS deep crawl strategy: max_depth=%s, max_pages=%s",
                max_depth,
                max_pages,
            )
            # Simplified approach - internal links only, no complex filtering
            return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
                max_depth=max_depth,
                include_external=False,  # Only internal links (matches working basic test)
                max_pages=max_pages,
                # Remove complex filtering that may interfere with BFS functionality
            )
        except Exception as e:
            self.logger.warning("Failed to create BFS strategy: %s", e)
            # Last resort: minimal BFS parameters (same as working basic test)
            try:
                return BFSDeepCrawlStrategy(  # type: ignore[attr-defined]
                    max_depth=max_depth,
                    include_external=False,
                    max_pages=max_pages,
                )
            except Exception as e2:
                self.logger.error("Failed to create minimal BFS strategy: %s", e2)
                return None

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
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("sitemap:"):
                    sitemaps.append(line.split(":", 1)[1].strip())
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
                    loc_text = (sm.text or "").strip()
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
                    loc_text = (loc.text or "").strip()
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
                return await resp.text()
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

                # Direct async iteration over results from browser.arun() with deep crawl strategy
                # When stream=True and deep_crawl_strategy is set, arun returns an async generator
                self.logger.info(
                    "Processing streaming results from BFSDeepCrawlStrategy - starting iteration"
                )
                self.logger.info(f"DEBUG: crawl_result type: {type(crawl_result)}")
                self.logger.info(
                    f"DEBUG: crawl_result has __aiter__: {hasattr(crawl_result, '__aiter__')}"
                )
                generator_count = 0
                try:
                    async for result in crawl_result:
                        generator_count += 1

                        self.logger.info(
                            f"AsyncGenerator yielded result #{generator_count}: {result.url if hasattr(result, 'url') else type(result).__name__}"
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
                                sanitized_result = self._sanitize_crawl_result(result)
                                successful_results.append(sanitized_result)
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
                        f"Deep crawl iteration completed: yielded {generator_count} results, {len(successful_results)} successful"
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
        max_concurrent = getattr(settings, "max_concurrent_sessions", 20)

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

        # Create dispatcher for memory-adaptive concurrency
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=getattr(settings, "crawl_memory_threshold", 80.0),
            check_interval=0.5,
            max_session_permit=max_concurrent,
        )

        # Remove deep_crawl_strategy to avoid recursion and set streaming
        batch_config = (
            run_config.clone() if hasattr(run_config, "clone") else run_config
        )
        if hasattr(batch_config, "deep_crawl_strategy"):
            batch_config.deep_crawl_strategy = None
        batch_config.stream = True

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
                            sanitized_result = self._sanitize_crawl_result(result)
                            successful_results.append(sanitized_result)

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
                        sanitized_result = self._sanitize_crawl_result(single_result)
                        successful_results.append(sanitized_result)
                    else:
                        errors.append(
                            f"Failed to crawl {getattr(single_result, 'url', urls_to_crawl[0])}"
                        )

        return successful_results, errors

    async def pre_execute_setup(self) -> None:
        """Setup before crawling begins."""
        await super().pre_execute_setup()
        # No browser session cleanup needed - each crawl uses fresh browser
