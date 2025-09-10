"""
Parallel crawling engine for optimized high-performance web crawler.

This module implements high-performance parallel crawling using crawl4ai's arun_many()
method with proper content validation and hash placeholder detection.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from dataclasses import dataclass

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.models import CrawlResult

from crawler_mcp.optimized_config import OptimizedConfig
from crawler_mcp.utils.monitoring import PerformanceMonitor  # for type reference

# ConcurrencyTuner removed - no longer using undocumented APIs

# Use only documented Crawl4AI APIs - no private imports


@dataclass
class CrawlStats:
    """Statistics for a crawling session"""

    urls_requested: int
    urls_successful: int
    urls_failed: int
    hash_placeholders: int
    total_duration: float
    pages_per_second: float
    total_content_length: int
    average_content_length: float


class ParallelEngine:
    """High-performance parallel crawling engine using arun_many()"""

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize parallel crawling engine.

        Args:
            config: Optional optimized crawler configuration
        """
        self.config = config or OptimizedConfig()
        self.logger = logging.getLogger(__name__)

    async def crawl_batch(
        self,
        urls: list[str],
        browser_config: BrowserConfig,
        crawler_config: CrawlerRunConfig,
        dispatcher=None,
        monitor: PerformanceMonitor | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[CrawlResult]:
        """
        Crawl multiple URLs in parallel using arun_many().

        This is the main method for high-performance parallel crawling that uses
        crawl4ai's arun_many() with proper content validation.

        Args:
            urls: List of URLs to crawl
            browser_config: Browser configuration
            crawler_config: Crawler run configuration
            dispatcher: Optional dispatcher for concurrency management
            progress_callback: Optional callback for progress updates

        Returns:
            List of successful CrawlResult objects
        """
        if not urls:
            return []

        start_time = time.time()
        successful_results = []
        failed_urls = []
        hash_placeholder_urls = []
        # Track URLs that had HTTP success but failed content validation
        http_successful_urls = []

        self.logger.info(f"Starting parallel crawl of {len(urls)} URLs")

        # Build a safe batch config from the caller's config so timeouts/caching
        # and extraction options actually apply, while keeping filters disabled.
        batch_config = self._prepare_batch_config(crawler_config)

        try:
            async with self._create_crawler(text_only=False) as crawler:
                results_processed = 0

                # Use arun_many for parallel processing
                results_generator = await crawler.arun_many(
                    urls=urls, config=batch_config, dispatcher=dispatcher
                )

                # Process results using unified processing logic
                async def process_result_iterator():
                    """Unified iterator for both streaming and batch modes"""
                    if hasattr(results_generator, "__aiter__"):
                        # Streaming mode - iterate over async generator
                        async for result in results_generator:
                            yield result
                    else:
                        # Batch mode - iterate over list
                        for result in results_generator:
                            yield result

                async for result in process_result_iterator():
                    results_processed += 1

                    if result.success:
                        # Track HTTP successful URLs regardless of content validation
                        http_successful_urls.append(result.url)

                        # Validate content quality
                        if self._is_valid_content(result, monitor):
                            successful_results.append(result)

                            # Report progress if callback provided
                            if progress_callback:
                                progress_callback(
                                    len(successful_results),
                                    len(urls),
                                    f"Crawled: {result.url}",
                                )
                        else:
                            # result.success is True but content deemed invalid
                            hash_placeholder_urls.append(result.url)
                            if monitor is not None:
                                import contextlib

                                with contextlib.suppress(Exception):
                                    monitor.record_hash_placeholder(result.url)
                            self.logger.warning(
                                f"Filtered invalid content: {result.url}"
                            )
                    else:
                        # result.success is False
                        failed_urls.append(result.url)
                        self.logger.debug(
                            f"Crawl failed: {result.url} - "
                            f"{getattr(result, 'error', 'Unknown error')}"
                        )

                    # Log progress periodically
                    if results_processed % 10 == 0:
                        self.logger.info(
                            f"Processed {results_processed}/{len(urls)} URLs"
                        )

        except Exception as e:
            self.logger.error(f"Parallel crawl failed: {e}")
            # Return whatever results we managed to get

        # Log final statistics
        end_time = time.time()
        duration = end_time - start_time
        pages_per_second = len(successful_results) / duration if duration > 0 else 0

        self.logger.info(
            f"Parallel crawl completed: {len(successful_results)} successful, "
            f"{len(failed_urls)} failed, "
            f"{len(hash_placeholder_urls)} hash placeholders, "
            f"{len(http_successful_urls)} HTTP successful, "
            f"{duration:.1f}s, {pages_per_second:.2f} pages/sec"
        )

        # Consider returning metadata separately (e.g., alongside results) if
        # needed downstream.

        return successful_results

    def _create_crawler(self, text_only: bool = False) -> AsyncWebCrawler:
        """Create crawler with appropriate configuration.

        Args:
            text_only: If True, use text_mode for lightweight crawling

        Returns:
            AsyncWebCrawler configured appropriately
        """
        if text_only:
            # Use text_mode for lightweight crawling (replaces HTTP strategy)
            config = BrowserConfig(text_mode=True, headless=True)
        else:
            # Standard browser with full features
            config = BrowserConfig(headless=True)

        return AsyncWebCrawler(config=config)

    async def crawl_batch_raw(
        self,
        urls: list[str],
        crawler_config: CrawlerRunConfig,
        dispatcher=None,
        text_only: bool = False,
    ) -> list[CrawlResult]:
        """
        Crawl multiple URLs and return raw CrawlResult objects without content
        validation.

        This is useful for discovery passes where link extraction is needed even if
        the page content is minimal.
        """
        if not urls:
            return []

        results: list[CrawlResult] = []

        batch_config = self._prepare_batch_config(crawler_config)

        try:
            async with self._create_crawler(text_only=False) as crawler:
                gen = await crawler.arun_many(
                    urls=urls, config=batch_config, dispatcher=dispatcher
                )
                if hasattr(gen, "__aiter__"):
                    async for r in gen:
                        if getattr(r, "success", False):
                            results.append(r)
                else:
                    for r in gen:
                        if getattr(r, "success", False):
                            results.append(r)
        except Exception as e:
            self.logger.debug(f"crawl_batch_raw failed: {e}")
        return results

    async def crawl_streaming(
        self,
        urls: list[str],
        browser_config: BrowserConfig,
        crawler_config: CrawlerRunConfig,
        dispatcher=None,
        monitor: PerformanceMonitor | None = None,
        result_callback: Callable[[CrawlResult], None] | None = None,
    ) -> AsyncGenerator[CrawlResult, None]:
        """
        Stream crawl results as they become available.

        This method yields results as soon as they are available, enabling
        real-time processing of crawled content.

        Args:
            urls: List of URLs to crawl
            browser_config: Browser configuration
            crawler_config: Crawler run configuration
            dispatcher: Optional dispatcher for concurrency management
            result_callback: Optional callback for each result

        Yields:
            CrawlResult objects as they become available
        """
        if not urls:
            return

        self.logger.info(f"Starting streaming crawl of {len(urls)} URLs")

        # Enable streaming in config
        batch_config = self._prepare_batch_config(crawler_config)
        batch_config.stream = True

        try:
            async with self._create_crawler(text_only=False) as crawler:
                results_generator = await crawler.arun_many(
                    urls=urls, config=batch_config, dispatcher=dispatcher
                )

                # Check if we get async generator (streaming) or list (batch)
                if hasattr(results_generator, "__aiter__"):
                    async for result in results_generator:
                        if result.success and self._is_valid_content(result, monitor):
                            # Call result callback if provided
                            if result_callback:
                                result_callback(result)

                            yield result
                        else:
                            if monitor is not None:
                                import contextlib

                                with contextlib.suppress(Exception):
                                    monitor.record_hash_placeholder(result.url)
                            self.logger.debug(f"Skipping invalid result: {result.url}")
                else:
                    # Batch mode - iterate over list
                    for result in results_generator:
                        if result.success and self._is_valid_content(result, monitor):
                            # Call result callback if provided
                            if result_callback:
                                result_callback(result)

                            yield result
                        else:
                            if monitor is not None:
                                import contextlib

                                with contextlib.suppress(Exception):
                                    monitor.record_hash_placeholder(result.url)
                            self.logger.debug(f"Skipping invalid result: {result.url}")

        except Exception as e:
            self.logger.error(f"Streaming crawl failed: {e}")

    async def crawl_batched(
        self,
        urls: list[str],
        browser_config: BrowserConfig,
        crawler_config: CrawlerRunConfig,
        dispatcher=None,
        batch_size: int = 50,
    ) -> list[CrawlResult]:
        """
        Crawl URLs in smaller batches to manage memory and resources.

        Args:
            urls: List of URLs to crawl
            browser_config: Browser configuration
            crawler_config: Crawler run configuration
            dispatcher: Optional dispatcher for concurrency management
            batch_size: Number of URLs per batch

        Returns:
            List of successful CrawlResult objects from all batches
        """
        if not urls:
            return []

        all_results = []
        total_batches = (len(urls) + batch_size - 1) // batch_size

        self.logger.info(
            f"Crawling {len(urls)} URLs in {total_batches} batches of {batch_size}"
        )

        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            self.logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_urls)} URLs)"
            )

            try:
                batch_results = await self.crawl_batch(
                    batch_urls, browser_config, crawler_config, dispatcher
                )
                all_results.extend(batch_results)

                self.logger.info(
                    f"Batch {batch_num} completed: {len(batch_results)} "
                    f"successful results"
                )

                # Brief pause between batches to prevent overwhelming the system
                if i + batch_size < len(urls):  # Don't sleep after last batch
                    await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Batch {batch_num} failed: {e}")
                # Continue with next batch even if current batch fails

        return all_results

    def _prepare_batch_config(
        self, crawler_config: CrawlerRunConfig
    ) -> CrawlerRunConfig:
        """
        Prepare crawler configuration optimized for batch processing.

        Args:
            crawler_config: Base crawler configuration

        Returns:
            Optimized crawler configuration for batch processing
        """
        # Clone the configuration to avoid modifying the original
        batch_config = self._clone_config(crawler_config)

        # Optimize for batch processing without breaking JS-rendered pages
        # Preserve or raise delay when JS rendering is enabled; keep moderate
        # delay otherwise.
        try:
            current_delay = float(
                getattr(batch_config, "delay_before_return_html", 0.5)
            )
        except Exception:
            current_delay = 0.5

        # Infer JS mode by policy rather than undocumented flags; caller/browser
        # config governs JS.
        js_mode = False

        with suppress(Exception):
            batch_config.delay_before_return_html = (
                max(current_delay, 3.0) if js_mode else max(current_delay, 0.5)
            )

        # Ensure proper content extraction settings
        if hasattr(batch_config, "verbose"):
            with suppress(Exception):
                batch_config.verbose = False  # Reduce log noise

        return batch_config

    def _clone_config(self, config: CrawlerRunConfig) -> CrawlerRunConfig:
        """Create a safe copy of crawler configuration using only documented fields."""
        try:
            excluded = (
                config.excluded_tags.copy()
                if getattr(config, "excluded_tags", None)
                else []
            )
        except Exception:
            excluded = []

        # Core documented fields safe to copy to constructor
        rc = CrawlerRunConfig(
            markdown_generator=getattr(config, "markdown_generator", None),
            excluded_tags=excluded,
            exclude_external_links=getattr(config, "exclude_external_links", True),
            cache_mode=getattr(config, "cache_mode", None),
            check_robots_txt=getattr(config, "check_robots_txt", False),
            word_count_threshold=getattr(config, "word_count_threshold", 50),
            page_timeout=getattr(config, "page_timeout", 30000),
        )

        # Crawl4AI-documented stable fields safe to copy via setattr
        documented_fields = [
            "delay_before_return_html",
            "wait_for",  # CSS/JS selectors for readiness
            "js_code",  # JavaScript code execution
            "process_iframes",  # Process iframe content
            "css_selector",  # CSS selector for content
            "stream",  # Enable streaming mode
        ]

        # Copy documented fields
        for name in documented_fields:
            try:
                if hasattr(config, name):
                    setattr(rc, name, getattr(config, name))
            except Exception as e:
                self.logger.debug(f"Failed to copy documented field {name}: {e}")

        return rc

    def _is_valid_content(
        self, result: CrawlResult, monitor: PerformanceMonitor | None = None
    ) -> bool:
        """
        Validate crawl result content quality.

        This method detects hash placeholders and ensures content meets
        minimum quality standards.

        Args:
            result: CrawlResult to validate

        Returns:
            True if content is valid, False if hash placeholder or low quality
        """
        if not result.success:
            return False

        try:
            # Extract content for validation with fallbacks
            content = self._extract_content_for_validation(result)

            if not content:
                # Log diagnostic info for empty content
                self.logger.debug(
                    f"Empty content for {getattr(result, 'url', 'unknown')}: "
                    f"markdown={hasattr(result, 'markdown')}, "
                    f"html={hasattr(result, 'html')}, "
                    f"extracted_content={hasattr(result, 'extracted_content')}"
                )
                return False

            # If validation is disabled, apply only minimal invalid checks
            if not self.config.content_validation:
                text = content.strip()
                if not text:
                    return False
                return not (len(text) == 32 and text.isalnum())

            # Validate content; return False if invalid for any reason
            reason = self._invalid_reason(content)
            if reason is None:
                if self.config.content_validation:
                    return self._validate_content_quality(content)
                return True
            # Potentially relax for documentation sections if configured
            try:
                url = getattr(result, "url", "")
                patterns = getattr(self.config, "doc_relax_validation_patterns", [])
                if url and patterns and any(re.search(p, url) for p in patterns):
                    if monitor is not None:
                        monitor.record_relaxed_acceptance(reason)
                    self.logger.debug(
                        f"Relaxed acceptance for {url} due to doc rule "
                        f"(reason={reason})"
                    )
                    return True
            except Exception:
                pass
            if monitor is not None:
                try:
                    monitor.record_invalid_reason(reason, getattr(result, "url", ""))
                except Exception:
                    monitor.record_invalid_reason(reason)
            # Log reason for debugging. The strategy will mark failures.
            self.logger.debug(
                f"Invalid content for {getattr(result, 'url', '')}: {reason}"
            )
            return False

        except Exception as e:
            self.logger.debug(f"Content validation failed for {result.url}: {e}")
            return False

    def _extract_content_for_validation(self, result: CrawlResult) -> str:
        """Extract content from result for validation with multiple fallbacks"""
        try:
            # Try markdown extraction first (original approach)
            if hasattr(result, "markdown") and result.markdown:
                if (
                    hasattr(result.markdown, "fit_markdown")
                    and result.markdown.fit_markdown
                ):
                    content = str(result.markdown.fit_markdown).strip()
                    if content:
                        return content
                elif (
                    hasattr(result.markdown, "raw_markdown")
                    and result.markdown.raw_markdown
                ):
                    content = str(result.markdown.raw_markdown).strip()
                    if content:
                        return content
                else:
                    content = str(result.markdown).strip()
                    if content:
                        return content

            # Fallback 1: Try extracted_content attribute
            if hasattr(result, "extracted_content") and result.extracted_content:
                content = str(result.extracted_content).strip()
                if content:
                    self.logger.debug(
                        f"Using extracted_content fallback for "
                        f"{getattr(result, 'url', 'unknown')}"
                    )
                    return content

            # Fallback 2: Try basic HTML text extraction
            if hasattr(result, "html") and result.html:
                import re

                html = str(result.html)
                # Remove script and style elements
                html = re.sub(
                    r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE
                )
                html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
                # Remove HTML tags
                text = re.sub(r"<[^>]+>", " ", html)
                # Clean whitespace
                text = re.sub(r"\s+", " ", text).strip()
                if text and len(text) > 50:  # Minimum threshold for HTML text
                    self.logger.debug(
                        f"Using HTML text fallback for "
                        f"{getattr(result, 'url', 'unknown')}"
                    )
                    return text

            # Fallback 3: Try any text-like attribute
            for attr in ["text", "content", "cleaned_html"]:
                if hasattr(result, attr):
                    content = getattr(result, attr)
                    if content and isinstance(content, str):
                        content = content.strip()
                        if content:
                            self.logger.debug(
                                f"Using {attr} fallback for "
                                f"{getattr(result, 'url', 'unknown')}"
                            )
                            return content

        except Exception as e:
            self.logger.debug(
                f"Content extraction failed for "
                f"{getattr(result, 'url', 'unknown')}: {e}"
            )

        return ""

    def _validate_content_quality(self, content: str) -> bool:
        """
        Perform additional content quality validation.

        Args:
            content: Content to validate

        Returns:
            True if content passes quality checks
        """
        # Check for reasonable text patterns
        if len(content) < 10:
            return False

        # Check for reasonable word-to-character ratio
        words = content.split()
        if len(words) > 0:
            avg_word_length = len(content.replace(" ", "")) / len(words)
            if (
                avg_word_length < 2 or avg_word_length > 20
            ):  # Suspiciously short/long words
                return False

        # Check for reasonable sentence structure (basic heuristic)
        sentence_endings = content.count(".") + content.count("!") + content.count("?")
        # Long text with no sentences is suspicious
        return not (len(words) > 50 and sentence_endings == 0)

    def _invalid_reason(self, content: str) -> str | None:
        """Return reason string if content is invalid, otherwise None."""
        text = content.strip()
        if not text:
            return "empty"

        words = text.split()
        if len(words) <= 1:
            return "very_short"

        # Check for common JavaScript framework loading states (whitelist)
        js_loading_patterns = [
            "loading",
            "please wait",
            "initializing",
            "loading...",
            "react",
            "vue",
            "angular",
            "next.js",
            "nuxt",
            "svelte",
            "app is loading",
            "javascript required",
            "enabling javascript",
        ]
        text_lower = text.lower()
        if (
            any(pattern in text_lower for pattern in js_loading_patterns)
            and len(text) < 200
            and len(words) < 30
        ):
            # This might be a JS loading state, not a hash placeholder
            # Still very minimal, could be legitimate loading state
            return None

        # More lenient hash-like detection - only flag if very specific patterns
        if (
            (len(text) in (32, 40, 64))
            and text.isalnum()
            and not any(
                word in text_lower
                for word in ["the", "and", "for", "with", "this", "that"]
            )
        ):
            # Additional check: real hash placeholders usually have no spaces or
            # common words
            return f"hash_like{len(text)}"

        # Lines comprised only of many '#' - but allow some markdown
        try:
            first_line = text.splitlines()[0].strip()
            if len(first_line) >= 12 and set(first_line) == {
                "#"
            }:  # Increased threshold
                return "hash_wall"
        except Exception:
            pass

        # Allow Markdown headings (leading '#'), but catch explicit placeholders
        if text.startswith("hash:"):
            return "hash_prefix"

        # More lenient word count threshold for JavaScript sites
        min_words = max(5, self.config.min_word_count // 8)  # Much more lenient
        if len(words) < min_words:
            return "too_few_words"

        # Suspicious average word length
        try:
            avg_word_length = len(text.replace(" ", "")) / len(words)
            if avg_word_length < 2 or avg_word_length > 20:
                return "avg_word_length"
        except Exception:
            pass

        # Relax sentence ending requirement; only flag extremely long blocks with none
        sentence_endings = text.count(".") + text.count("!") + text.count("?")
        if len(words) > 200 and sentence_endings == 0:
            return "no_sentence_endings"

        return None

    def calculate_stats(
        self, results: list[CrawlResult], duration: float, total_requested: int
    ) -> CrawlStats:
        """
        Calculate statistics for a crawling session.

        Args:
            results: List of successful results
            duration: Total crawling duration in seconds
            total_requested: Total number of URLs requested

        Returns:
            CrawlStats object with session statistics
        """
        successful = len(results)
        failed = total_requested - successful

        # Calculate content statistics
        total_content_length = 0
        hash_placeholders = 0

        for result in results:
            content = self._extract_content_for_validation(result)
            content_length = len(content)
            total_content_length += content_length

            # Check for hash placeholders in successful results (shouldn't happen)
            if content_length <= 32 and content.strip().isalnum():
                hash_placeholders += 1

        pages_per_second = successful / duration if duration > 0 else 0
        avg_content_length = total_content_length / successful if successful > 0 else 0

        return CrawlStats(
            urls_requested=total_requested,
            urls_successful=successful,
            urls_failed=failed,
            hash_placeholders=hash_placeholders,
            total_duration=duration,
            pages_per_second=pages_per_second,
            total_content_length=total_content_length,
            average_content_length=avg_content_length,
        )
