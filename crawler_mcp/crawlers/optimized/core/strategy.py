"""
Main orchestrator strategy for optimized high-performance web crawler.

This module implements the main crawler strategy that inherits from Crawl4AI's
AsyncCrawlerStrategy and orchestrates all components for maximum performance.
"""

import asyncio
import logging
import math
import os
import re
import time
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from crawl4ai.async_crawler_strategy import AsyncCrawlerStrategy
from crawl4ai.models import AsyncCrawlResponse

from ....models.crawl import PageContent
from ..clients.github_client import GitHubClient
from ..clients.qdrant_http_client import QdrantClient
from ..clients.tei_client import TEIEmbeddingsClient
from ..config import OptimizedConfig
from ..factories.browser_factory import BrowserFactory
from ..factories.content_extractor import ContentExtractorFactory
from ..factories.dispatcher_factory import DispatcherFactory
from ..processing.result_converter import ResultConverter
from ..processing.url_discovery import URLDiscovery
from ..utils.monitoring import PerformanceMonitor
from .parallel_engine import ParallelEngine


class OptimizedCrawlerStrategy(AsyncCrawlerStrategy):
    """
    High-performance web crawler strategy using Crawl4AI patterns.

    This strategy inherits from AsyncCrawlerStrategy to be compatible with
    the Crawl4AI ecosystem while providing enhanced performance and features.
    """

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize optimized crawler strategy.

        Args:
            config: Optional configuration (defaults to OptimizedConfig())
        """
        super().__init__()

        self.config = config or OptimizedConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize all components
        self.url_discovery = URLDiscovery(self.config)
        self.browser_factory = BrowserFactory(self.config)
        self.content_extractor = ContentExtractorFactory(self.config)
        self.dispatcher_factory = DispatcherFactory(self.config)
        self.parallel_engine = ParallelEngine(self.config)
        self.result_converter = ResultConverter(self.config)
        self.monitor = PerformanceMonitor(self.config)

        # Internal state
        self._session_active = False
        self._last_pr_report: dict[str, Any] | None = None

        self.logger.info(
            f"Initialized OptimizedCrawlerStrategy with config: {self.config}"
        )

    async def crawl(self, url: str, **kwargs) -> AsyncCrawlResponse:
        """
        Main crawl method required by AsyncCrawlerStrategy.

        This method implements the complete optimized crawling pipeline:
        1. URL Discovery from sitemaps
        2. Parallel crawling with arun_many()
        3. Content validation and conversion
        4. Performance monitoring

        Args:
            url: Starting URL to crawl from
            **kwargs: Additional crawling parameters

        Returns:
            AsyncCrawlResponse compatible with Crawl4AI
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting optimized crawl from: {url}")

            # Parse crawling parameters
            max_urls = kwargs.get("max_urls", self.config.max_urls_to_discover)
            max_concurrent = kwargs.get(
                "max_concurrent", self.config.max_concurrent_crawls
            )

            # Special mode: GitHub PR URL -> fetch via GitHub API and synthesize page
            try:
                if self._is_github_pr_url(url):
                    return await self._crawl_github_pr(url, start_time)
            except Exception as e:
                self.logger.warning(
                    f"GitHub PR fast-path failed, falling back to normal crawl: {e}"
                )

            # Phase 1: URL Discovery
            self.logger.info("Phase 1: Starting URL discovery")

            # If max_urls is 1, skip discovery and just use the provided URL
            if max_urls == 1:
                self.logger.info(
                    "Single URL mode - skipping discovery, using provided URL"
                )
                discovered_urls = [url]
            else:
                discovered_urls = await self._discover_urls(url, max_urls)

                if not discovered_urls:
                    self.logger.warning(
                        f"No URLs discovered for {url}, using original URL"
                    )
                    discovered_urls = [url]

            # Phase 2: Setup crawling infrastructure
            self.logger.info("Phase 2: Setting up crawling infrastructure")
            (
                browser_config,
                crawler_config,
                dispatcher,
            ) = await self._setup_infrastructure(max_concurrent)

            # Phase 3: Start monitoring
            self.monitor.start_crawl_monitoring(discovered_urls)

            # Phase 4: Parallel crawling
            self.logger.info(
                f"Phase 4: Starting parallel crawl of {len(discovered_urls)} URLs"
            )
            crawl_results = await self._execute_parallel_crawl(
                discovered_urls,
                browser_config,
                crawler_config,
                dispatcher,
                stream=bool(kwargs.get("stream", False)),
            )

            # Phase 5: Result processing and conversion
            self.logger.info("Phase 5: Processing and converting results")
            response = await self._process_results(
                crawl_results, discovered_urls, start_time, url
            )

            # Phase 6: Finalize monitoring
            final_metrics = self.monitor.finish_crawl_monitoring()

            # Log final statistics
            duration = time.time() - start_time
            self.logger.info(
                f"Optimized crawl completed: {len(crawl_results)} pages in {duration:.1f}s "
                f"({final_metrics.pages_per_second:.2f} pages/sec)"
            )

            return response

        except Exception as e:
            self.logger.error(f"Optimized crawl failed for {url}: {e}", exc_info=True)
            # Return minimal response on error
            return await self._create_error_response(url, str(e))

    async def start(self) -> None:
        """
        Initialize crawler resources.

        This method is called by Crawl4AI to initialize the strategy.
        """
        if self._session_active:
            return

        self.logger.info("Starting optimized crawler strategy")

        try:
            # Initialize URL discovery session
            await self.url_discovery.__aenter__()

            self._session_active = True
            self.logger.info("Optimized crawler strategy started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start crawler strategy: {e}")
            raise

    async def close(self) -> None:
        """
        Clean up crawler resources.

        This method is called by Crawl4AI to clean up the strategy.
        """
        if not self._session_active:
            return

        self.logger.info("Closing optimized crawler strategy")

        try:
            # Close URL discovery session
            await self.url_discovery.__aexit__(None, None, None)

            self._session_active = False
            self.logger.info("Optimized crawler strategy closed successfully")

        except Exception as e:
            self.logger.warning(f"Error during crawler strategy cleanup: {e}")

    def set_hook(self, hook_type: str, hook: Callable) -> None:
        """
        Set custom hook for monitoring events.

        Args:
            hook_type: Type of hook to set
            hook: Hook function to register
        """
        try:
            self.monitor.register_hook(hook_type, hook)
            self.logger.debug(f"Registered hook: {hook_type}")
        except ValueError as e:
            self.logger.warning(f"Failed to register hook {hook_type}: {e}")

    async def _discover_urls(self, start_url: str, max_urls: int) -> list[str]:
        """
        Discover URLs from the starting URL.

        Args:
            start_url: URL to start discovery from
            max_urls: Maximum URLs to discover

        Returns:
            List of discovered URLs
        """
        try:
            # Ensure URL discovery is ready
            if (
                not hasattr(self.url_discovery, "_session")
                or self.url_discovery._session is None
            ):
                await self.url_discovery._ensure_session()

            # Discover URLs
            discovered_urls = await self.url_discovery.discover_all(start_url, max_urls)

            self.logger.info(f"Discovered {len(discovered_urls)} URLs from {start_url}")

            # Enhanced fallback: trigger if sitemap discovery yields too few quality URLs
            fallback_threshold = self._calculate_fallback_threshold(max_urls)
            quality_urls = self._assess_url_quality(discovered_urls, start_url)

            should_trigger_fallback = (
                getattr(self.config, "fallback_link_discovery", True)
                and quality_urls < fallback_threshold
                and max_urls > 1
            )

            if should_trigger_fallback:
                self.logger.info(
                    f"🔄 Triggering fallback discovery: found {quality_urls}/{len(discovered_urls)} quality URLs "
                    f"(threshold: {fallback_threshold})"
                )
                try:
                    # Build a small JS-enabled fetch to extract internal links from the start page
                    from urllib.parse import urljoin, urlparse

                    js_needed = bool(getattr(self.config, "fallback_require_js", True))
                    if js_needed:
                        fb_browser = self.browser_factory.create_javascript_config()
                    else:
                        fb_browser = self.browser_factory.get_recommended_config()

                    fb_run_cfg = self.content_extractor.create_crawler_config()
                    if hasattr(fb_run_cfg, "enable_javascript"):
                        fb_run_cfg.enable_javascript = True
                    if hasattr(fb_run_cfg, "wait_for_js_rendering"):
                        fb_run_cfg.wait_for_js_rendering = True
                    if hasattr(fb_run_cfg, "delay_before_return_html"):
                        fb_run_cfg.delay_before_return_html = max(
                            0.5, getattr(fb_run_cfg, "delay_before_return_html", 0.0)
                        )

                    # Single fetch
                    first = await self.parallel_engine.crawl_batch_raw(
                        [start_url], fb_browser, fb_run_cfg, None
                    )
                    if first:
                        r = first[0]
                        base_host = urlparse(start_url).netloc
                        # Use the same link extraction logic as reporting to avoid schema mismatches
                        try:
                            candidates = self.result_converter._extract_links(r)  # type: ignore[arg-type]
                        except Exception:
                            links = getattr(r, "links", None)
                            candidates = []
                            if isinstance(links, dict):
                                candidates = list(links.get("internal", []) or [])
                            elif isinstance(links, list):
                                candidates = links[:]
                        # Normalize and filter to same host
                        out: list[str] = []
                        seen: set[str] = {start_url}
                        for lk in candidates:
                            try:
                                pu = urlparse(lk)
                                if not pu.netloc:
                                    abs_url = urljoin(start_url, lk)
                                else:
                                    abs_url = lk
                                # Locale filtering if configured
                                ok_locale = True
                                allowed_locales = [
                                    s.lower()
                                    for s in (
                                        getattr(self.config, "allowed_locales", [])
                                        or []
                                    )
                                ]
                                if allowed_locales:
                                    p = urlparse(abs_url)
                                    segs = [s for s in p.path.split("/") if s]
                                    loc = segs[0].lower() if segs else ""
                                    if not loc:
                                        ok_locale = ("" in allowed_locales) or (
                                            "en" in allowed_locales
                                        )
                                    else:
                                        ok_locale = (loc in allowed_locales) or any(
                                            a and (loc == a or loc.startswith(a + "-"))
                                            for a in allowed_locales
                                        )
                                if (
                                    urlparse(abs_url).netloc == base_host
                                    and ok_locale
                                    and abs_url not in seen
                                ):
                                    seen.add(abs_url)
                                    out.append(abs_url)
                            except Exception:
                                continue
                        # Apply budget and deduplicate (normalize trailing slash)
                        budget = max(
                            1, int(getattr(self.config, "fallback_max_links", 200))
                        )
                        candidates = [start_url, *out[: min(budget, max_urls - 1)]]

                        # dedupe with normalization
                        def _norm(u: str) -> str:
                            try:
                                pu = urlparse(u)
                                path = pu.path or "/"
                                if not path.endswith("/") and "." not in (
                                    path.split("/")[-1] or ""
                                ):
                                    path = path + "/"
                                return urljoin(f"{pu.scheme}://{pu.netloc}", path)
                            except Exception:
                                return u

                        seen_norm: set[str] = set()
                        expanded: list[str] = []
                        for u in candidates:
                            nu = _norm(u)
                            if nu not in seen_norm:
                                seen_norm.add(nu)
                                expanded.append(nu)
                        if len(expanded) > len(discovered_urls):
                            original_count = len(discovered_urls)
                            discovered_urls = expanded
                            self.logger.info(
                                f"🔄 Fallback discovery: {original_count} → {len(expanded)} URLs (+{len(expanded) - original_count} internal links)"
                            )
                        else:
                            self.logger.warning(
                                "🚫 Fallback discovery didn't improve URL count"
                            )
                except Exception as e:
                    self.logger.warning(f"⚠️ Fallback URL discovery failed: {e}")

            # Trigger monitoring hook
            await self.monitor.trigger_hook(
                "url_discovered", start_url=start_url, discovered_urls=discovered_urls
            )

            return discovered_urls

        except Exception as e:
            self.logger.error(f"URL discovery failed for {start_url}: {e}")
            return [start_url]  # Fallback to original URL

    def _calculate_fallback_threshold(self, max_urls: int) -> int:
        """Calculate minimum URLs needed before triggering fallback discovery."""
        # Use configured ratio and absolute minimum
        ratio = getattr(self.config, "fallback_min_quality_ratio", 0.1)
        absolute_min = getattr(self.config, "fallback_absolute_minimum", 3)

        # Calculate threshold based on configuration
        ratio_based = max(1, int(max_urls * ratio))
        return max(absolute_min, ratio_based)

    def _assess_url_quality(self, urls: list[str], start_url: str) -> int:
        """Assess the quality of discovered URLs and return count of quality URLs."""
        if not urls:
            return 0

        quality_count = 0
        start_domain = urlparse(start_url).netloc

        for url in urls:
            try:
                parsed = urlparse(url)

                # Must be same domain
                if parsed.netloc != start_domain:
                    continue

                # Skip if it's just the homepage
                if url == start_url or parsed.path in ("/", "/index.html", "/home"):
                    continue

                # Skip obvious non-content URLs
                path_lower = parsed.path.lower()
                if any(
                    skip in path_lower
                    for skip in [
                        "/api/",
                        "/admin/",
                        "/login/",
                        "/logout/",
                        "/register/",
                        "/search/",
                        "/contact/",
                        "/privacy/",
                        "/terms/",
                        ".css",
                        ".js",
                        ".xml",
                        ".txt",
                        ".json",
                    ]
                ):
                    continue

                # Count as quality URL
                quality_count += 1

            except Exception:
                continue

        self.logger.debug(
            f"🔍 Quality assessment: {quality_count}/{len(urls)} URLs are quality content"
        )
        return quality_count

    async def _setup_infrastructure(self, max_concurrent: int) -> tuple:
        """
        Setup crawling infrastructure (browser, config, dispatcher).

        Args:
            max_concurrent: Maximum concurrent sessions

        Returns:
            Tuple of (browser_config, crawler_config, dispatcher)
        """
        # Create browser configuration
        browser_config = self.browser_factory.get_recommended_config()

        # Create content extraction configuration
        markdown_gen = self.content_extractor.create_markdown_generator()
        crawler_config = self.content_extractor.create_crawler_config(markdown_gen)

        # Create dispatcher for concurrency management
        dispatcher = self.dispatcher_factory.get_recommended_dispatcher()

        self.logger.debug(
            f"Infrastructure setup: {max_concurrent} concurrent sessions, "
            f"robots_txt={crawler_config.check_robots_txt}"
        )

        return browser_config, crawler_config, dispatcher

    async def _execute_parallel_crawl(
        self,
        urls: list[str],
        browser_config,
        crawler_config,
        dispatcher,
        *,
        stream: bool = False,
    ) -> list[Any]:
        """
        Execute parallel crawling of URLs.

        Args:
            urls: URLs to crawl
            browser_config: Browser configuration
            crawler_config: Crawler configuration
            dispatcher: Concurrency dispatcher

        Returns:
            List of successful crawl results
        """

        # Create progress callback for monitoring
        def progress_callback(completed: int, total: int, current_url: str):
            if completed % 10 == 0:  # Log every 10 pages
                self.logger.info(f"Progress: {completed}/{total} pages completed")

        # Execute parallel crawl
        if stream:
            # Stream results and accumulate
            results = []
            async for r in self.parallel_engine.crawl_streaming(
                urls,
                browser_config,
                crawler_config,
                dispatcher,
                monitor=self.monitor,
                result_callback=None,
            ):
                results.append(r)
        elif self.config.use_aggressive_mode:
            # Use retry logic for aggressive mode
            results = await self.parallel_engine.crawl_with_retry(
                urls,
                browser_config,
                crawler_config,
                dispatcher,
                monitor=self.monitor,
                max_retries=1,  # Limited retries for speed
            )
        else:
            # Standard parallel crawl
            results = await self.parallel_engine.crawl_batch(
                urls,
                browser_config,
                crawler_config,
                dispatcher,
                monitor=self.monitor,
                progress_callback=progress_callback,
            )

        # Optional JS-only retry for failed URLs to recover JS-heavy pages
        try:
            successful_urls = {getattr(r, "url", "") for r in results}
            failed_urls = [u for u in urls if u not in successful_urls]
            # Only retry if there are failures and default config is JS-disabled
            if (
                failed_urls
                and not self.config.javascript_enabled
                and getattr(self.config, "js_retry_enabled", True)
            ):
                self.logger.info(
                    f"Retrying {len(failed_urls)} URLs with JavaScript enabled"
                )
                js_browser_config = self.browser_factory.create_javascript_config()
                js_results = await self.parallel_engine.crawl_batch(
                    failed_urls,
                    js_browser_config,
                    crawler_config,  # keep run settings minimal; engine uses None internally
                    dispatcher,
                    monitor=self.monitor,
                    progress_callback=progress_callback,
                )
                results.extend(js_results)
        except Exception as e:
            self.logger.warning(f"JS retry pass skipped due to error: {e}")

        # Record results in monitoring
        for result in results:
            if hasattr(result, "success") and result.success:
                content_text = self.result_converter._extract_markdown_content(result)
                content_length = len(content_text)
                crawl_time = getattr(result, "crawl_time", 0.0)
                self.monitor.record_page_success(result.url, content_length, crawl_time)
                # Record a simple quality score based on content length (0..1)
                try:
                    quality_score = min(1.0, max(0.0, content_length / 5000.0))
                    is_dup = self.monitor.record_content_hash(result.url, content_text)
                    if is_dup:
                        # also increment duplicate metric (record_content_validation adds when flagged)
                        pass
                    self.monitor.record_content_validation(
                        result.url, quality_score, is_duplicate=is_dup
                    )
                except Exception:
                    pass
            else:
                error_msg = getattr(result, "error", "Unknown error")
                self.monitor.record_page_failure(result.url, str(error_msg))

        # Also mark URLs that did not return a successful result as failures
        try:
            successful_urls = {
                getattr(r, "url", "") for r in results if getattr(r, "success", False)
            }
            remaining_failures = [u for u in urls if u not in successful_urls]
            for u in remaining_failures:
                self.monitor.record_page_failure(u, "failed_after_retries")
        except Exception:
            pass

        # Fallback: if discovery produced too few URLs, expand from in-page links
        try:
            if (
                getattr(self.config, "fallback_link_discovery", True)
                and len(results) <= 2
                and len(urls) <= 2
            ):
                # Collect internal links from successful pages
                from urllib.parse import urljoin, urlparse

                base_host = urlparse(urls[0]).netloc if urls else ""
                base_root = f"{urlparse(urls[0]).scheme}://{base_host}" if urls else ""
                link_pool: list[str] = []
                seen: set[str] = set(urls)
                for r in results:
                    if getattr(r, "success", False):
                        try:
                            links = getattr(r, "links", None)
                            if isinstance(links, dict):
                                candidates = links.get("internal", []) or []
                            else:
                                candidates = links or []
                            for lk in candidates:
                                try:
                                    pu = urlparse(lk)
                                    # Treat relative URLs as internal; resolve against page URL or site root
                                    if not pu.netloc:
                                        abs_url = urljoin(
                                            getattr(r, "url", base_root) or base_root,
                                            lk,
                                        )
                                        if abs_url and abs_url not in seen:
                                            seen.add(abs_url)
                                            link_pool.append(abs_url)
                                    elif pu.netloc == base_host:
                                        if lk not in seen:
                                            seen.add(lk)
                                            link_pool.append(lk)
                                except Exception:
                                    continue
                        except Exception:
                            continue

                if link_pool:
                    # Limit to fallback budget and avoid duplicates
                    budget = max(
                        1, int(getattr(self.config, "fallback_max_links", 200))
                    )
                    to_crawl = link_pool[:budget]
                    self.logger.info(
                        f"Fallback link discovery: scheduling {len(to_crawl)} internal links"
                    )
                    # Choose JS-enabled browser if requested, as docs/nav often require it
                    if (not self.config.javascript_enabled) and bool(
                        getattr(self.config, "fallback_require_js", True)
                    ):
                        fb_browser = self.browser_factory.create_javascript_config()
                    else:
                        fb_browser = browser_config
                    # Use a JS-friendly run config for fallback to handle SPA/docs navigation
                    try:
                        fb_run_cfg = self.content_extractor.create_crawler_config()
                        if hasattr(fb_run_cfg, "enable_javascript"):
                            fb_run_cfg.enable_javascript = True
                        if hasattr(fb_run_cfg, "delay_before_return_html"):
                            fb_run_cfg.delay_before_return_html = max(
                                0.5,
                                getattr(fb_run_cfg, "delay_before_return_html", 0.0),
                            )
                        if hasattr(fb_run_cfg, "wait_for_js_rendering"):
                            fb_run_cfg.wait_for_js_rendering = True
                        if hasattr(fb_run_cfg, "process_iframes"):
                            fb_run_cfg.process_iframes = False
                        if hasattr(fb_run_cfg, "only_text"):
                            fb_run_cfg.only_text = True
                    except Exception:
                        fb_run_cfg = crawler_config

                    fb_results = await self.parallel_engine.crawl_batch(
                        to_crawl,
                        fb_browser,
                        fb_run_cfg,
                        dispatcher,
                        monitor=self.monitor,
                        progress_callback=progress_callback,
                    )
                    # Append and record
                    for fr in fb_results:
                        results.append(fr)
                        if getattr(fr, "success", False):
                            try:
                                content_text = (
                                    self.result_converter._extract_markdown_content(fr)
                                )
                                content_length = len(content_text)
                                crawl_time = getattr(fr, "crawl_time", 0.0)
                                self.monitor.record_page_success(
                                    fr.url, content_length, crawl_time
                                )
                                qs = min(1.0, max(0.0, content_length / 5000.0))
                                is_dup = self.monitor.record_content_hash(
                                    fr.url, content_text
                                )
                                self.monitor.record_content_validation(
                                    fr.url, qs, is_duplicate=is_dup
                                )
                            except Exception:
                                pass
                        else:
                            self.monitor.record_page_failure(
                                fr.url, getattr(fr, "error", "failed")
                            )
        except Exception as e:
            self.logger.debug(f"Fallback link discovery skipped due to error: {e}")

        # Follow internal links BFS pass (bounded), useful when sitemaps are minimal
        try:
            if getattr(self.config, "follow_internal_links", True):
                from urllib.parse import urljoin, urlparse

                base_hosts = {urlparse(u).netloc for u in urls}
                # Harvest internal links from successful pages
                link_set: list[str] = []
                seen: set[str] = set(urls)
                budget = max(
                    0, int(getattr(self.config, "follow_internal_budget", 200))
                )
                allowed_locales = [
                    s.lower()
                    for s in (getattr(self.config, "allowed_locales", []) or [])
                ]

                def _ok_locale(abs_url: str) -> bool:
                    if not allowed_locales:
                        return True
                    try:
                        p = urlparse(abs_url)
                        segs = [s for s in p.path.split("/") if s]
                        loc = segs[0].lower() if segs else ""
                        if not loc:
                            return ("" in allowed_locales) or ("en" in allowed_locales)
                        return (loc in allowed_locales) or any(
                            a and (loc == a or loc.startswith(a + "-"))
                            for a in allowed_locales
                        )
                    except Exception:
                        return True

                def _norm(u: str) -> str:
                    try:
                        pu = urlparse(u)
                        path = pu.path or "/"
                        if not path.endswith("/") and "." not in (
                            path.split("/")[-1] or ""
                        ):
                            path = path + "/"
                        return urljoin(f"{pu.scheme}://{pu.netloc}", path)
                    except Exception:
                        return u

                for r in results:
                    if getattr(r, "success", False):
                        try:
                            # Prefer converter to extract robust link lists
                            try:
                                candidates = self.result_converter._extract_links(r)  # type: ignore[arg-type]
                            except Exception:
                                links = getattr(r, "links", None)
                                candidates = []
                                if isinstance(links, dict):
                                    candidates = list(links.get("internal", []) or [])
                                elif isinstance(links, list):
                                    candidates = links[:]
                            # host not needed directly; base_hosts already computed
                            for lk in candidates:
                                try:
                                    pu = urlparse(lk)
                                    absu = (
                                        lk
                                        if pu.netloc
                                        else urljoin(getattr(r, "url", ""), lk)
                                    )
                                    if urlparse(absu).netloc in base_hosts:
                                        nu = _norm(absu)
                                        if _ok_locale(nu) and nu not in seen:
                                            seen.add(nu)
                                            link_set.append(nu)
                                            if len(link_set) >= budget:
                                                break
                                except Exception:
                                    continue
                            if len(link_set) >= budget:
                                break
                        except Exception:
                            continue
                if link_set:
                    self.logger.info(
                        f"Follow-internal pass: scheduling {len(link_set)} internal links"
                    )
                    fb_browser = browser_config
                    if not self.config.javascript_enabled:
                        import contextlib

                        with contextlib.suppress(Exception):
                            fb_browser = self.browser_factory.create_javascript_config()
                    fb_results = await self.parallel_engine.crawl_batch(
                        link_set,
                        fb_browser,
                        crawler_config,
                        dispatcher,
                        monitor=self.monitor,
                        progress_callback=progress_callback,
                    )
                    results.extend(fb_results)
        except Exception as e:
            self.logger.debug(f"Follow-internal pass skipped due to error: {e}")

        return results

    def _is_github_pr_url(self, url: str) -> bool:
        try:
            return bool(
                re.match(r"^https://github\.com/[^/]+/[^/]+/pull/\d+(?:/.*)?$", url)
            )
        except Exception:
            return False

    def _parse_github_pr_parts(self, url: str) -> tuple[str, str, int]:
        m = re.match(r"^https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
        if not m:
            raise ValueError("Not a GitHub PR URL")
        owner, repo, num = m.group(1), m.group(2), int(m.group(3))
        return owner, repo, num

    async def _crawl_github_pr(self, url: str, start_time: float) -> AsyncCrawlResponse:
        owner, repo, number = self._parse_github_pr_parts(url)
        token = os.getenv("GITHUB_TOKEN", "")
        if not token:
            self.logger.warning(
                "GITHUB_TOKEN not set; GitHub API may rate-limit or fail"
            )

        pr: dict[str, Any]
        reviews: list[dict[str, Any]]
        rcomments: list[dict[str, Any]]
        icomments: list[dict[str, Any]]

        async with GitHubClient(token=token, timeout_s=20.0) as gh:
            pr = await gh.get_pull_request(owner, repo, number)
            # Reviews and comments (paginated)
            reviews = await gh.list_reviews(owner, repo, number)
            rcomments = await gh.list_review_comments(owner, repo, number)
            icomments = await gh.list_issue_comments(owner, repo, number)
            try:
                files = await gh.list_pull_files(owner, repo, number)
            except Exception:
                files = []

        title = pr.get("title", f"PR #{number}")
        state = pr.get("state", "")
        merged = bool(pr.get("merged", False))
        author = (pr.get("user", {}) or {}).get("login", "")
        created_at = pr.get("created_at", "")
        updated_at = pr.get("updated_at", "")
        body = pr.get("body", "") or ""

        # Build per-item pages for granular embeddings
        pages: list[PageContent] = []

        # PR overview page
        overview_lines: list[str] = []
        overview_lines.append(f"# {title} (#{number})")
        overview_lines.append(f"Repository: {owner}/{repo}")
        overview_lines.append(f"Author: {author}")
        overview_lines.append(f"State: {'merged' if merged else state}")
        if created_at:
            overview_lines.append(f"Created: {created_at}")
        if updated_at:
            overview_lines.append(f"Updated: {updated_at}")
        overview_lines.append("")
        if body:
            overview_lines.append("## PR Description")
            overview_lines.append(body)
            overview_lines.append("")
        overview_md = "\n".join(overview_lines).strip()
        pages.append(
            PageContent(
                url=url,
                title=f"PR #{number}: {title}",
                content=overview_md,
                markdown=overview_md,
                html="",
                links=[],
                images=[],
                metadata={
                    "source": "github_pr",
                    "item_type": "pr_overview",
                    "owner": owner,
                    "repo": repo,
                    "pr_number": number,
                    "pr_state": state,
                    "pr_merged": merged,
                    "author": author,
                    "reviews_count": len(reviews),
                    "review_comments_count": len(rcomments),
                    "issue_comments_count": len(icomments),
                },
            )
        )

        # Review items
        for rv in reviews:
            reviewer = (rv.get("user", {}) or {}).get("login", "")
            r_state = rv.get("state", "")
            submitted = rv.get("submitted_at", rv.get("commit_id", ""))
            r_body = rv.get("body", "") or ""
            review_id = rv.get("id")
            review_url = (
                rv.get("html_url") or f"{url}#pullrequestreview-{review_id}"
                if review_id
                else url
            )

            lines_r: list[str] = []
            lines_r.append(f"## Review by {reviewer} - {r_state}")
            if submitted:
                lines_r.append(f"Submitted: {submitted}")
            lines_r.append("")
            if r_body:
                lines_r.append(r_body)
            md_r = "\n".join(lines_r).strip()
            pages.append(
                PageContent(
                    url=review_url,
                    title=f"PR #{number} Review: {reviewer} - {r_state}",
                    content=md_r,
                    markdown=md_r,
                    html="",
                    links=[],
                    images=[],
                    metadata={
                        "source": "github_pr",
                        "item_type": "pr_review",
                        "owner": owner,
                        "repo": repo,
                        "pr_number": number,
                        "author": reviewer,
                        "review_id": review_id,
                        "review_state": r_state,
                        "submitted_at": submitted,
                    },
                )
            )

        # Review comments (code)
        for c in rcomments:
            c_user = (c.get("user", {}) or {}).get("login", "")
            path = c.get("path", "")
            line_no = c.get("line") or c.get("original_line") or ""
            created = c.get("created_at", "")
            body_c = c.get("body", "") or ""
            comment_id = c.get("id")
            comment_url = c.get("html_url") or (
                f"{url}#discussion_r{comment_id}" if comment_id else url
            )

            lines_c: list[str] = []
            header = f"## Review Comment by {c_user}"
            if path:
                header += f" on {path}:{line_no}"
            lines_c.append(header)
            if created:
                lines_c.append(f"Created: {created}")
            lines_c.append("")
            if body_c:
                lines_c.append(body_c)
            md_c = "\n".join(lines_c).strip()
            pages.append(
                PageContent(
                    url=comment_url,
                    title=f"PR #{number} Review Comment: {path}:{line_no}",
                    content=md_c,
                    markdown=md_c,
                    html="",
                    links=[],
                    images=[],
                    metadata={
                        "source": "github_pr",
                        "item_type": "pr_review_comment",
                        "owner": owner,
                        "repo": repo,
                        "pr_number": number,
                        "author": c_user,
                        "comment_id": comment_id,
                        "path": path,
                        "line": line_no,
                        "created_at": created,
                    },
                )
            )

        # Conversation comments (on PR thread)
        for c in icomments:
            c_user = (c.get("user", {}) or {}).get("login", "")
            created = c.get("created_at", "")
            body_c = c.get("body", "") or ""
            comment_id = c.get("id")
            comment_url = c.get("html_url") or url

            lines_ic: list[str] = []
            lines_ic.append(f"## Conversation Comment by {c_user}")
            if created:
                lines_ic.append(f"Created: {created}")
            lines_ic.append("")
            if body_c:
                lines_ic.append(body_c)
            md_ic = "\n".join(lines_ic).strip()
            pages.append(
                PageContent(
                    url=comment_url,
                    title=f"PR #{number} Conversation Comment by {c_user}",
                    content=md_ic,
                    markdown=md_ic,
                    html="",
                    links=[],
                    images=[],
                    metadata={
                        "source": "github_pr",
                        "item_type": "pr_conversation_comment",
                        "owner": owner,
                        "repo": repo,
                        "pr_number": number,
                        "author": c_user,
                        "comment_id": comment_id,
                        "created_at": created,
                    },
                )
            )

        # Build a structured PR report for downstream reporting
        try:
            reviewer_states: dict[str, int] = {}
            reviewers: set[str] = set()
            for rv in reviews:
                st = str(rv.get("state", "")).upper()
                reviewer_states[st] = reviewer_states.get(st, 0) + 1
                u = (rv.get("user", {}) or {}).get("login")
                if u:
                    reviewers.add(str(u))

            participants: set[str] = set()
            if author:
                participants.add(author)
            for c in rcomments:
                u = (c.get("user", {}) or {}).get("login")
                if u:
                    participants.add(str(u))
            for c in icomments:
                u = (c.get("user", {}) or {}).get("login")
                if u:
                    participants.add(str(u))

            comments_per_file: dict[str, int] = {}
            for c in rcomments:
                pth = str(c.get("path", ""))
                if pth:
                    comments_per_file[pth] = comments_per_file.get(pth, 0) + 1

            file_stats = []
            for f in files or []:
                p = str(f.get("filename", ""))
                file_stats.append(
                    {
                        "path": p,
                        "status": f.get("status"),
                        "additions": f.get("additions"),
                        "deletions": f.get("deletions"),
                        "changes": f.get("changes"),
                        "comments": comments_per_file.get(p, 0),
                    }
                )

            self._last_pr_report = {
                "owner": owner,
                "repo": repo,
                "pr_number": number,
                "title": title,
                "state": state,
                "merged": merged,
                "author": author,
                "created_at": created_at,
                "updated_at": updated_at,
                "reviews_total": len(reviews),
                "review_states": reviewer_states,
                "reviewers": sorted(reviewers),
                "review_comments_total": len(rcomments),
                "conversation_comments_total": len(icomments),
                "participants": sorted(participants),
                "files_total": len(files or []),
                "files": file_stats,
            }
        except Exception:
            self._last_pr_report = None

        # Attach embeddings for all items (and upsert to Qdrant if enabled)
        try:
            if self.config.enable_embeddings:

                class _Tmp:
                    def __init__(self, pages_list):
                        self.pages = pages_list

                await self._attach_embeddings(_Tmp(pages))
        except Exception as e:  # pragma: no cover
            self.logger.warning(
                f"TEI embeddings skipped for GitHub PR due to error: {e}"
            )

        # Build response similar to _process_results, with one synthesized page
        combined_md = "\n\n".join(
            [getattr(pg, "markdown", "") or getattr(pg, "content", "") for pg in pages]
        )
        combined_html = (
            f"<!-- EXTRACTED_CONTENT -->\n{combined_md}\n<!-- END_CONTENT -->\n"
        )
        response = AsyncCrawlResponse(
            html=combined_html, status_code=200, response_headers={}
        )
        response.response_headers["X-Crawl-Success"] = "True"
        response.response_headers["X-Crawl-URL"] = url
        response.response_headers["X-Pages-Crawled"] = "1"
        response.response_headers["X-URLs-Discovered"] = "1"
        response.response_headers["X-Pages-Per-Second"] = "1.0"
        response.response_headers["X-Placeholders-Filtered"] = str(
            self.monitor.metrics.hash_placeholders_detected
        )
        response.response_headers["X-Total-Links"] = "0"
        response.response_headers["X-Extraction-Method"] = "github_pr_api"
        response.response_headers["X-Content-Length"] = str(len(combined_md))

        # Store last pages for downstream consumers
        try:
            self._last_pages = pages
        except Exception:
            self._last_pages = []

        return response

    def get_pr_report(self) -> dict[str, Any] | None:
        """Return the last GitHub PR report if available."""
        return self._last_pr_report

    async def _process_results(
        self,
        results: list[Any],
        original_urls: list[str],
        start_time: float,
        start_url: str,
    ) -> AsyncCrawlResponse:
        """
        Process crawl results and create response.

        Args:
            results: Crawl results from parallel engine
            original_urls: Original URLs that were requested
            start_time: Start time of crawl
            start_url: Original starting URL

        Returns:
            AsyncCrawlResponse for Crawl4AI compatibility
        """
        try:
            # Convert to our internal models first
            crawl_result = self.result_converter.batch_to_crawl_result(
                results, original_urls, start_time
            )
            try:
                self._last_pages = list(crawl_result.pages)
            except Exception:
                self._last_pages = []

            # Optionally enrich with embeddings via TEI
            try:
                if self.config.enable_embeddings and crawl_result.pages:
                    await self._attach_embeddings(crawl_result)
            except Exception as e:  # pragma: no cover
                self.logger.warning(f"TEI embeddings skipped due to error: {e}")

            # Create AsyncCrawlResponse compatible with Crawl4AI
            # Note: This is a simplified conversion - you may need to adjust
            # based on the exact AsyncCrawlResponse structure

            # Combine all content
            combined_content = ""
            combined_html = ""
            all_links = []

            for page in crawl_result.pages:
                combined_content += page.content + "\n\n"
                combined_html += page.html + "\n"
                all_links.extend(page.links)

            # Create response object - only use valid AsyncCrawlResponse fields
            response = AsyncCrawlResponse(
                html=combined_html.strip(),
                status_code=200 if results else 500,
                response_headers={},
            )

            # Store custom data in response_headers since we can't add fields
            response.response_headers["X-Crawl-Success"] = str(len(results) > 0)
            response.response_headers["X-Crawl-URL"] = start_url
            response.response_headers["X-Pages-Crawled"] = str(len(results))
            response.response_headers["X-URLs-Discovered"] = str(len(original_urls))
            response.response_headers["X-Pages-Per-Second"] = str(
                crawl_result.statistics.pages_per_second
            )
            response.response_headers["X-Placeholders-Filtered"] = str(
                self.monitor.metrics.hash_placeholders_detected
            )
            # Failed URL diagnostics
            try:
                successful_urls = {getattr(r, "url", "") for r in results}
                failed_urls = [u for u in original_urls if u not in successful_urls]
                response.response_headers["X-Failed-URLs-Count"] = str(len(failed_urls))
                if failed_urls:
                    sample = ",".join(failed_urls[:5])
                    response.response_headers["X-Failed-URLs-Sample"] = sample
            except Exception:
                response.response_headers["X-Failed-URLs-Count"] = "0"

            # Aggregate link sample and totals
            try:
                seen: set[str] = set()
                sample: list[str] = []
                for page in crawl_result.pages:
                    for link in page.links:
                        if link not in seen:
                            seen.add(link)
                            if len(sample) < 5:
                                sample.append(link)
                response.response_headers["X-Total-Links"] = str(len(seen))
                if sample:
                    response.response_headers["X-Sample-Links"] = ",".join(sample)
            except Exception:
                pass

            # Human-readable content sizes
            try:
                total_bytes = len(combined_content.encode("utf-8"))

                def _fmt_bytes(n: int | float) -> str:
                    n = float(n)
                    units = ["B", "KB", "MB", "GB"]
                    i = 0
                    while n >= 1024 and i < len(units) - 1:
                        n /= 1024.0
                        i += 1
                    return f"{n:.2f} {units[i]}"

                avg_size = total_bytes / max(1, len(results))
                response.response_headers["X-Total-Content-Human"] = _fmt_bytes(
                    total_bytes
                )
                response.response_headers["X-Average-Page-Size-Human"] = _fmt_bytes(
                    avg_size
                )
            except Exception:
                pass

            response.response_headers["X-Extraction-Method"] = "optimized_parallel"
            response.response_headers["X-Content-Length"] = str(len(combined_content))

            # Always prepend extracted content to HTML for retrieval
            if combined_content:
                content_section = f"<!-- EXTRACTED_CONTENT -->\n{combined_content}\n<!-- END_CONTENT -->\n\n"
                response.html = content_section + (response.html or "")

            return response

        except Exception as e:
            self.logger.error(f"Failed to process results: {e}")
            return await self._create_error_response(start_url, str(e))

    async def _attach_embeddings(self, crawl_result) -> None:
        """Generate embeddings for each page content via TEI in batches."""
        pages = getattr(crawl_result, "pages", []) or []
        if not pages:
            return

        endpoint = getattr(self.config, "tei_endpoint", "http://localhost:8080")
        # Base batch size from config; we may adapt it below for better parallelism
        cfg_batch = max(1, int(getattr(self.config, "tei_batch_size", 16)))
        timeout_s = float(getattr(self.config, "tei_timeout_s", 15.0))
        retries = int(getattr(self.config, "tei_max_retries", 1))
        model = getattr(self.config, "tei_model_name", "")

        # Collect inputs with light preprocessing to control token load
        texts: list[str] = []
        idxs: list[int] = []
        # If tei_max_input_chars <= 0, do not truncate inputs
        max_chars = int(getattr(self.config, "tei_max_input_chars", 4000) or 0)
        collapse_ws = bool(getattr(self.config, "tei_collapse_whitespace", True))
        for i, p in enumerate(pages):
            try:
                raw = getattr(p, "content", None) or ""
                if not raw:
                    continue
                txt = raw.strip()
                if collapse_ws:
                    # simple whitespace collapse
                    txt = " ".join(txt.split())
                if max_chars > 0 and len(txt) > max_chars:
                    txt = txt[:max_chars]
                if not txt:
                    continue
                texts.append(txt)
                idxs.append(i)
            except Exception:
                continue

        if not texts:
            return

        parallel = max(1, int(getattr(self.config, "tei_parallel_requests", 4)))
        # Respect server cap if provided
        srv_cap = int(getattr(self.config, "tei_max_concurrent_requests", 0) or 0)
        if srv_cap > 0:
            parallel = min(parallel, srv_cap)

        # Build length-aware batches to better utilize GPU throughput
        # Token-aware target based on server limits
        max_batch_tokens = int(getattr(self.config, "tei_max_batch_tokens", 0) or 0)
        chars_per_tok = float(getattr(self.config, "tei_chars_per_token", 4.0) or 4.0)
        target_chars = max(
            4000,
            int(getattr(self.config, "tei_target_chars_per_batch", 64000) or 64000),
        )
        if max_batch_tokens > 0 and chars_per_tok > 0:
            target_chars = int(max_batch_tokens * chars_per_tok)

        max_items = max(
            1,
            min(
                int(getattr(self.config, "tei_max_client_batch_size", 128) or 128),
                cfg_batch,
            ),
        )

        pairs = list(zip(idxs, texts, strict=False))
        # Sort by length (desc) for greedy packing
        pairs.sort(key=lambda it: len(it[1]), reverse=True)
        batches: list[list[tuple[int, str]]] = []
        cur: list[tuple[int, str]] = []
        cur_chars = 0
        for pi, tx in pairs:
            tlen = len(tx)
            # If adding this would exceed limits, flush current batch
            if cur and (cur_chars + tlen > target_chars or len(cur) >= max_items):
                batches.append(cur)
                cur = []
                cur_chars = 0
            cur.append((pi, tx))
            cur_chars += tlen
        if cur:
            batches.append(cur)

        # Ensure at least `parallel` batches when possible to keep workers busy
        # by splitting the largest batches until we reach desired count.
        while len(batches) < parallel and any(len(b) > 1 for b in batches):
            # find largest by chars
            li = max(
                range(len(batches)), key=lambda i: sum(len(t) for _, t in batches[i])
            )
            big = batches.pop(li)
            mid = len(big) // 2
            batches.append(big[:mid])
            batches.append(big[mid:])

        # Fallback if no batches created
        if not batches and texts:
            batches = [[(i, t)] for i, t in pairs]

        async with TEIEmbeddingsClient(
            endpoint, model=model or None, timeout_s=timeout_s, max_retries=retries
        ) as client:
            sem = asyncio.Semaphore(parallel)
            tasks: list[asyncio.Task] = []

            async def worker(
                assign_idxs: list[int], chunk: list[str]
            ) -> tuple[list[int], list[list[float]], float, int]:
                async with sem:
                    t0 = time.time()
                    vecs = await client.embed_texts(chunk)
                    dt = (time.time() - t0) * 1000.0
                    return assign_idxs, vecs, dt, len(chunk)

            # Launch variable-size batches
            for batch_items in batches:
                b_idxs = [pi for (pi, _) in batch_items]
                b_texts = [tx for (_, tx) in batch_items]
                tasks.append(asyncio.create_task(worker(b_idxs, b_texts)))

            # Assign results as they complete
            total_batches = 0
            total_ms = 0.0
            total_items = 0
            dim_sample = None
            for fut in asyncio.as_completed(tasks):
                try:
                    assign_idxs, vecs, t_ms, n_items = await fut
                    total_batches += 1
                    total_ms += float(t_ms)
                    total_items += int(n_items)
                    for off, vec in enumerate(vecs):
                        try:
                            pi = assign_idxs[off]
                            meta = getattr(pages[pi], "metadata", None)
                            if isinstance(meta, dict):
                                # Optional projection to target dim
                                tdim = int(
                                    getattr(self.config, "embedding_target_dim", 0) or 0
                                )
                                proj = str(
                                    getattr(self.config, "embedding_projection", "none")
                                    or "none"
                                )
                                v = vec
                                if tdim > 0 and isinstance(v, list) and len(v) != tdim:
                                    if len(v) > tdim and proj == "truncate":
                                        v = v[:tdim]
                                    elif len(v) < tdim and proj == "pad_zero":
                                        v = v + [0.0] * (tdim - len(v))
                                meta["embedding"] = v
                                meta["embedding_model"] = model or "tei"
                                meta["embedding_provider"] = "hf-tei"
                                if dim_sample is None and isinstance(v, list):
                                    dim_sample = len(v)
                        except Exception:
                            continue
                except Exception as e:
                    self.logger.debug(f"TEI batch failed: {e}")

            # Attach a compact summary to the monitor report
            try:
                avg_latency = (total_ms / total_batches) if total_batches else 0.0
                # Report average batch size used
                avg_batch = (total_items / total_batches) if total_batches else 0.0
                self.monitor.record_embeddings_stats(
                    endpoint=endpoint,
                    model=model or "tei",
                    pages=len(texts),
                    batches=total_batches,
                    batch_size=round(avg_batch, 1),
                    parallel_requests=parallel,
                    avg_batch_latency_ms=round(avg_latency, 1),
                    vector_dim=int(dim_sample or 0),
                )
            except Exception:
                pass

        # Optional: upsert to Qdrant if enabled
        try:
            if getattr(self.config, "enable_qdrant", False):
                dim = int(
                    getattr(self.config, "qdrant_vector_size", 0) or (dim_sample or 0)
                )
                if dim <= 0:
                    # Cannot upsert without knowing vector dimension
                    self.logger.warning(
                        "Qdrant upsert skipped: unknown vector dimension"
                    )
                else:
                    await self._upsert_qdrant(crawl_result, vector_dim=dim)
        except Exception as e:
            self.logger.warning(f"Qdrant upsert skipped due to error: {e}")

    async def _upsert_qdrant(self, crawl_result, *, vector_dim: int) -> None:
        pages = getattr(crawl_result, "pages", []) or []
        if not pages:
            return
        url = getattr(self.config, "qdrant_url", "http://localhost:6333")
        collection = getattr(self.config, "qdrant_collection", "crawler_pages")
        distance = getattr(self.config, "qdrant_distance", "Cosine")
        vectors_name = getattr(self.config, "qdrant_vectors_name", "") or None
        cfg_batch = max(1, int(getattr(self.config, "qdrant_batch_size", 128)))
        parallel = max(1, int(getattr(self.config, "qdrant_parallel_requests", 2)))
        wait = bool(getattr(self.config, "qdrant_upsert_wait", True))
        api_key = getattr(self.config, "qdrant_api_key", None)

        # Build points
        points: list[dict] = []
        for p in pages:
            try:
                meta = getattr(p, "metadata", {}) or {}
                vec = meta.get("embedding")
                if not isinstance(vec, list):
                    continue
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, getattr(p, "url", "")))
                payload = {
                    "url": getattr(p, "url", ""),
                    "title": getattr(p, "title", ""),
                    "word_count": getattr(p, "word_count", 0),
                    "timestamp": str(getattr(p, "timestamp", "")),
                }
                # Include content (bounded) for retrieval
                content = getattr(p, "content", "") or ""
                if content:
                    payload["text"] = content[:10000]
                # Merge other metadata (shallow)
                for k, v in meta.items() if isinstance(meta, dict) else []:
                    if k == "embedding":
                        continue
                    if k not in payload:
                        payload[k] = v
                point = {"id": pid, "vector": vec, "payload": payload}
                if vectors_name:
                    point = {
                        "id": pid,
                        "vector": {vectors_name: vec},
                        "payload": payload,
                    }
                points.append(point)
            except Exception:
                continue

        if not points:
            return

        async with QdrantClient(url, api_key=api_key, timeout_s=20.0) as qc:
            # Ensure collection exists
            await qc.ensure_collection(
                collection,
                size=vector_dim,
                distance=distance,
                vectors_name=vectors_name,
            )

            # Adapt batch size to ensure we create at least as many batches
            # as parallel workers when possible, capped by configured batch.
            if points:
                desired = max(1, math.ceil(len(points) / max(1, parallel)))
                batch = min(cfg_batch, desired)
            else:
                batch = cfg_batch

            sem = asyncio.Semaphore(parallel)
            tasks: list[asyncio.Task] = []

            async def worker(chunk_start: int, chunk: list[dict]):
                async with sem:
                    t0 = time.time()
                    await qc.upsert(collection, chunk, wait=wait)
                    return (time.time() - t0) * 1000.0, len(chunk)

            for i in range(0, len(points), batch):
                tasks.append(asyncio.create_task(worker(i, points[i : i + batch])))

            batches = 0
            total_ms = 0.0
            total_pts = 0
            for fut in asyncio.as_completed(tasks):
                try:
                    ms, n = await fut
                    batches += 1
                    total_ms += float(ms)
                    total_pts += int(n)
                except Exception as e:
                    self.logger.debug(f"Qdrant upsert batch failed: {e}")

            try:
                avg_latency = (total_ms / batches) if batches else 0.0
                self.monitor.record_vectorstore_stats(
                    provider="qdrant",
                    url=url,
                    collection=collection,
                    points=total_pts,
                    batches=batches,
                    batch_size=batch,
                    parallel_requests=parallel,
                    avg_batch_latency_ms=round(avg_latency, 1),
                )
            except Exception:
                pass

    async def _create_error_response(self, url: str, error: str) -> AsyncCrawlResponse:
        """
        Create error response for failed crawls.

        Args:
            url: URL that failed
            error: Error message

        Returns:
            AsyncCrawlResponse indicating failure
        """
        response = AsyncCrawlResponse(
            html=f"<!-- ERROR: {error} -->", status_code=500, response_headers={}
        )

        # Store error info in headers
        response.response_headers["X-Crawl-Success"] = "False"
        response.response_headers["X-Crawl-URL"] = url
        response.response_headers["X-Crawl-Error"] = error
        response.response_headers["X-Extraction-Method"] = "optimized_failed"

        return response

    def get_performance_report(self) -> dict[str, Any]:
        """
        Get comprehensive performance report.

        Returns:
            Dictionary with performance metrics and recommendations
        """
        return self.monitor.get_performance_report()

    def get_last_pages(self) -> list[Any]:
        """Return the PageContent list from the last crawl, if available."""
        return self._last_pages

    def update_config(self, new_config: OptimizedConfig) -> None:
        """
        Update configuration and reinitialize components.

        Args:
            new_config: New configuration to apply
        """
        self.config = new_config

        # Reinitialize components with new config
        self.url_discovery = URLDiscovery(self.config)
        self.browser_factory = BrowserFactory(self.config)
        self.content_extractor = ContentExtractorFactory(self.config)
        self.dispatcher_factory = DispatcherFactory(self.config)
        self.parallel_engine = ParallelEngine(self.config)
        self.result_converter = ResultConverter(self.config)
        self.monitor = PerformanceMonitor(self.config)

        self.logger.info("Configuration updated and components reinitialized")

    async def crawl_single_url(self, url: str, **kwargs) -> dict[str, Any]:
        """
        Crawl a single URL using the optimized pipeline.

        This is a convenience method for single URL crawling that returns
        our internal model format instead of AsyncCrawlResponse.

        Args:
            url: URL to crawl
            **kwargs: Additional parameters

        Returns:
            Dictionary with crawl results in our format
        """
        # Force single URL mode
        kwargs["max_urls"] = 1

        start_time = time.time()

        try:
            # Use the main crawl method
            response = await self.crawl(url, **kwargs)

            # Convert back to our format for easier consumption
            headers = response.response_headers or {}
            success = headers.get("X-Crawl-Success", "False").lower() == "true"

            # Extract content from HTML if it contains extracted content marker
            content = ""
            if response.html and "<!-- EXTRACTED_CONTENT -->" in response.html:
                start_marker = "<!-- EXTRACTED_CONTENT -->\n"
                end_marker = "\n<!-- END_CONTENT -->"
                start_idx = response.html.find(start_marker)
                end_idx = response.html.find(end_marker)
                if start_idx != -1 and end_idx != -1:
                    content = response.html[start_idx + len(start_marker) : end_idx]

            return {
                "success": success,
                "url": headers.get("X-Crawl-URL", url),
                "content": content,
                "html": response.html,
                "metadata": {
                    "pages_crawled": int(headers.get("X-Pages-Crawled", "0")),
                    "urls_discovered": int(headers.get("X-URLs-Discovered", "0")),
                    "pages_per_second": float(headers.get("X-Pages-Per-Second", "0")),
                    "extraction_method": headers.get("X-Extraction-Method", "unknown"),
                    "content_length": int(headers.get("X-Content-Length", "0")),
                    "error": headers.get("X-Crawl-Error", ""),
                },
                "links": {"internal": [], "external": []},
                "duration": time.time() - start_time,
            }

        except Exception as e:
            return {
                "success": False,
                "url": url,
                "content": "",
                "html": "",
                "metadata": {"error": str(e)},
                "links": {"internal": [], "external": []},
                "duration": time.time() - start_time,
            }

    def __str__(self) -> str:
        """String representation of the strategy"""
        return (
            f"OptimizedCrawlerStrategy("
            f"concurrent={self.config.max_concurrent_crawls}, "
            f"max_urls={self.config.max_urls_to_discover}, "
            f"active={self._session_active})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the strategy"""
        return (
            f"OptimizedCrawlerStrategy(config={self.config}, "
            f"session_active={self._session_active})"
        )
