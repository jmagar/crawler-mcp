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
from dataclasses import dataclass
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.models import CrawlResult

from ..config import OptimizedConfig
from ..utils.monitoring import PerformanceMonitor  # for type reference
from .adaptive_dispatcher import ConcurrencyTuner


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

        self.logger.info(f"Starting parallel crawl of {len(urls)} URLs")

        # Build a safe batch config from the caller's config so timeouts/caching
        # and extraction options actually apply, while keeping filters disabled.
        batch_config = self._prepare_batch_config(crawler_config)

        tuner: ConcurrencyTuner | None = None

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Best-effort resource blocking via Playwright if available
                await self._enable_resource_blocking(crawler)

                # Start adaptive concurrency tuning if monitor and dispatcher are provided
                if monitor is not None and dispatcher is not None:
                    try:
                        max_conc = int(
                            getattr(
                                dispatcher,
                                "max_session_permit",
                                self.config.max_concurrent_crawls,
                            )
                        )
                    except Exception:
                        max_conc = self.config.max_concurrent_crawls
                    tuner = ConcurrencyTuner(
                        dispatcher=dispatcher,
                        monitor=monitor,
                        min_concurrency=2,
                        max_concurrency=max(2, max_conc),
                        sample_interval=2.0,
                    )
                    tuner.start()
                results_processed = 0

                # Use arun_many for parallel processing
                results_generator = await crawler.arun_many(
                    urls=urls, config=batch_config, dispatcher=dispatcher
                )

                # Check if we get a list (batch mode) or async generator (streaming mode)
                if hasattr(results_generator, "__aiter__"):
                    # Streaming mode - iterate over async generator
                    async for result in results_generator:
                        results_processed += 1

                        if result.success:
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
                                hash_placeholder_urls.append(result.url)
                                if monitor is not None:
                                    import contextlib

                                    with contextlib.suppress(Exception):
                                        monitor.record_hash_placeholder(result.url)
                                self.logger.warning(
                                    f"Filtered invalid content: {result.url}"
                                )
                        else:
                            failed_urls.append(result.url)
                            self.logger.debug(
                                f"Crawl failed: {result.url} - {getattr(result, 'error', 'Unknown error')}"
                            )

                        # Log progress periodically
                        if results_processed % 10 == 0:
                            self.logger.info(
                                f"Processed {results_processed}/{len(urls)} URLs"
                            )

                else:
                    # Batch mode - results_generator is a list
                    for result in results_generator:
                        results_processed += 1

                        if result.success:
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
                                f"Crawl failed: {result.url} - {getattr(result, 'error', 'Unknown error')}"
                            )

                        # Log progress periodically
                        if results_processed % 10 == 0:
                            self.logger.info(
                                f"Processed {results_processed}/{len(urls)} URLs"
                            )

        except Exception as e:
            self.logger.error(f"Parallel crawl failed: {e}")
            # Return whatever results we managed to get
        finally:
            if tuner:
                tuner.stop()

        # Log final statistics
        end_time = time.time()
        duration = end_time - start_time
        pages_per_second = len(successful_results) / duration if duration > 0 else 0

        self.logger.info(
            f"Parallel crawl completed: {len(successful_results)} successful, "
            f"{len(failed_urls)} failed, {len(hash_placeholder_urls)} hash placeholders, "
            f"{duration:.1f}s, {pages_per_second:.2f} pages/sec"
        )

        return successful_results

    async def crawl_batch_raw(
        self,
        urls: list[str],
        browser_config: BrowserConfig,
        crawler_config: CrawlerRunConfig,
        dispatcher=None,
    ) -> list[CrawlResult]:
        """
        Crawl multiple URLs and return raw CrawlResult objects without content validation.

        This is useful for discovery passes where link extraction is needed even if
        the page content is minimal.
        """
        if not urls:
            return []

        results: list[CrawlResult] = []

        batch_config = self._prepare_batch_config(crawler_config)

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                await self._enable_resource_blocking(crawler)
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

        tuner: ConcurrencyTuner | None = None

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                await self._enable_resource_blocking(crawler)
                if monitor is not None and dispatcher is not None:
                    try:
                        max_conc = int(
                            getattr(
                                dispatcher,
                                "max_session_permit",
                                self.config.max_concurrent_crawls,
                            )
                        )
                    except Exception:
                        max_conc = self.config.max_concurrent_crawls
                    tuner = ConcurrencyTuner(
                        dispatcher=dispatcher,
                        monitor=monitor,
                        min_concurrency=2,
                        max_concurrency=max(2, max_conc),
                        sample_interval=2.0,
                    )
                    tuner.start()
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
        finally:
            if tuner:
                tuner.stop()

    async def crawl_with_retry(
        self,
        urls: list[str],
        browser_config: BrowserConfig,
        crawler_config: CrawlerRunConfig,
        dispatcher=None,
        monitor: PerformanceMonitor | None = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> list[CrawlResult]:
        """
        Crawl URLs with automatic retry for failed requests.

        Args:
            urls: List of URLs to crawl
            browser_config: Browser configuration
            crawler_config: Crawler run configuration
            dispatcher: Optional dispatcher for concurrency management
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            List of successful CrawlResult objects
        """
        all_results = []
        urls_to_retry = urls.copy()

        for attempt in range(max_retries + 1):
            if not urls_to_retry:
                break

            attempt_results = await self.crawl_batch(
                urls_to_retry,
                browser_config,
                crawler_config,
                dispatcher,
                monitor=monitor,
            )
            # Track successful URLs
            successful_urls = {result.url for result in attempt_results}
            all_results.extend(attempt_results)

            # Determine which URLs need retry
            urls_to_retry = [u for u in urls_to_retry if u not in successful_urls]

            if urls_to_retry and attempt < max_retries:
                self.logger.info(
                    f"Retrying {len(urls_to_retry)} failed URLs (attempt {attempt + 2}/{max_retries + 1})"
                )
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)

        if urls_to_retry:
            self.logger.warning(
                f"Failed to crawl {len(urls_to_retry)} URLs after {max_retries + 1} attempts"
            )

        return all_results

    async def _enable_resource_blocking(self, crawler: Any) -> None:
        """
        Best-effort request interception to block heavy resources via Playwright if accessible.

        Tries to access underlying Page/Context to route requests and abort resource types
        like images/media/fonts/stylesheet and common ad/analytics URLs. No-op if unsupported.
        """
        try:
            # Try common attributes for context or page
            context = None
            for attr in ("context", "_context", "browser_context"):
                context = getattr(crawler, attr, None)
                if context:
                    break
            if context and hasattr(context, "route"):
                await context.route("**/*", self._route_handler)
                return

            page = getattr(crawler, "page", None) or getattr(crawler, "_page", None)
            if page and hasattr(page, "route"):
                await page.route("**/*", self._route_handler)
        except Exception:
            return

    async def _route_handler(self, route, request) -> None:  # type: ignore[no-redef]
        try:
            rtype = getattr(request, "resource_type", "")
            url = getattr(request, "url", "") or ""
            if rtype in {"image", "media", "font"} or any(
                bad in url
                for bad in (
                    "/ads",
                    "googletagmanager",
                    "doubleclick",
                    "facebook",
                    "pixel",
                )
            ):
                try:
                    await route.abort()
                except Exception:
                    await route.continue_()
            else:
                await route.continue_()
        except Exception:
            import contextlib

            with contextlib.suppress(Exception):
                await route.continue_()

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
                    f"Batch {batch_num} completed: {len(batch_results)} successful results"
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

        # Optimize for batch processing
        try:
            batch_config.delay_before_return_html = 0.5  # Moderate delay
        except Exception:
            pass

        # Ensure proper content extraction settings
        if hasattr(batch_config, "verbose"):
            try:
                batch_config.verbose = False  # Reduce log noise
            except Exception:
                pass

        return batch_config

    def _clone_config(self, config: CrawlerRunConfig) -> CrawlerRunConfig:
        """Create a cautious copy of crawler configuration.

        Only copies stable fields to avoid regressions across Crawl4AI versions.
        Content filtering remains disabled by construction in the generator.
        """
        try:
            excluded = (
                config.excluded_tags.copy()
                if getattr(config, "excluded_tags", None)
                else []
            )
        except Exception:
            excluded = []

        rc = CrawlerRunConfig(
            markdown_generator=getattr(config, "markdown_generator", None),
            excluded_tags=excluded,
            exclude_external_links=getattr(config, "exclude_external_links", True),
            cache_mode=getattr(
                config, "cache_mode", getattr(config, "cache_mode", None)
            ),
            check_robots_txt=getattr(config, "check_robots_txt", False),
            word_count_threshold=getattr(config, "word_count_threshold", 50),
            page_timeout=getattr(config, "page_timeout", 30000),
        )

        for name in (
            "only_text",
            "exclude_social_media_links",
            "process_iframes",
            "remove_overlay_elements",
            "delay_before_return_html",
            "enable_javascript",
            "wait_for_js_rendering",
        ):
            try:
                if hasattr(config, name):
                    setattr(rc, name, getattr(config, name))
            except Exception:
                pass

        if hasattr(rc, "verbose"):
            try:
                rc.verbose = False
            except Exception:
                pass

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

        # Check if markdown content exists
        if not hasattr(result, "markdown") or not result.markdown:
            return False

        try:
            # Extract content for validation
            content = self._extract_content_for_validation(result)

            if not content:
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
                        f"Relaxed acceptance for {url} due to doc rule (reason={reason})"
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
        """Extract content from result for validation"""
        try:
            if (
                hasattr(result.markdown, "fit_markdown")
                and result.markdown.fit_markdown
            ):
                return str(result.markdown.fit_markdown).strip()
            elif (
                hasattr(result.markdown, "raw_markdown")
                and result.markdown.raw_markdown
            ):
                return str(result.markdown.raw_markdown).strip()
            else:
                return str(result.markdown).strip()
        except Exception:
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

        # Hash-like placeholders
        if len(text) == 32 and text.isalnum():
            return "hash_like32"
        # Allow Markdown headings (leading '#'), but catch explicit placeholders
        if text.startswith("hash:"):
            return "hash_prefix"

        # Word count threshold (lenient)
        if len(words) < max(1, self.config.min_word_count // 5):
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
