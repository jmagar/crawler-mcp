"""
Slim crawl orchestrator that delegates to specialized managers and strategies.
Replaces the massive crawler_service.py with a clean, maintainable architecture.
"""

import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..config import settings
from ..crawlers import (
    DirectoryCrawlStrategy,
    DirectoryRequest,
    RepositoryCrawlStrategy,
    RepositoryRequest,
    WebCrawlStrategy,
)
from ..models.crawl import (
    CrawlRequest,
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from .logging import get_logger
from .memory import MemoryManager, cleanup_memory_manager, get_memory_manager
from .mixins import AsyncServiceBase
from .utils import suppress_stdout

logger = get_logger(__name__)


class CrawlerService(AsyncServiceBase):
    """
    Optimized crawler orchestrator with modular architecture.

    This replaces the 2189-line monolithic crawler_service.py with a clean,
    maintainable service that delegates to specialized components:
    - MemoryManager: 70% threshold with predictive cleanup
    - Strategies: Modular crawling for web, directory, repository
    - Direct AsyncWebCrawler usage without custom browser pooling
    """

    def __init__(self) -> None:
        super().__init__()

        # Managers (initialized lazily)
        self._memory_manager: MemoryManager | None = None

        # Strategies
        self._web_strategy = WebCrawlStrategy()
        self._directory_strategy = DirectoryCrawlStrategy()
        self._repository_strategy = RepositoryCrawlStrategy()

        # State
        self._initialized = False

    async def _initialize(self) -> None:
        """Service-specific initialization."""
        self.logger.info("Initializing crawler service with optimized components")

        # Initialize managers
        self._memory_manager = get_memory_manager()

        self.logger.info(
            f"Crawler service initialized - "
            f"Memory threshold: {settings.crawl_memory_threshold}%, "
            f"Direct AsyncWebCrawler usage enabled"
        )

    async def crawl_website(
        self,
        request: CrawlRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """
        Crawl a website using the optimized web strategy.

        Args:
            request: Web crawl request
            progress_callback: Optional progress reporting callback

        Returns:
            CrawlResult with crawled pages and statistics
        """
        await self.initialize()

        self.logger.info(f"Starting website crawl: {request.url}")

        # Validate request
        if not await self._web_strategy.validate_request(request):
            return CrawlResult(
                request_id=f"invalid_web_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=[request.url] if isinstance(request.url, str) else request.url,
                pages=[],
                errors=["Invalid web crawl request"],
                statistics=getattr(request, "statistics", CrawlStatistics()),
            )

        # Execute web crawling strategy
        return await self._web_strategy.execute(request, progress_callback)

    async def crawl_directory(
        self,
        directory_path: str,
        file_patterns: list[str] | None = None,
        recursive: bool = True,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """
        Crawl a local directory using the optimized directory strategy.

        Args:
            directory_path: Path to directory to crawl
            file_patterns: File patterns to include (e.g., ["*.py", "*.md"])
            recursive: Whether to crawl subdirectories
            progress_callback: Optional progress reporting callback

        Returns:
            CrawlResult with processed files and statistics
        """
        await self.initialize()

        self.logger.info(f"Starting directory crawl: {directory_path}")

        # Create directory request
        request = DirectoryRequest(
            directory_path=directory_path,
            file_patterns=file_patterns,
            recursive=recursive,
        )

        # Validate request
        if not await self._directory_strategy.validate_request(request):
            return CrawlResult(
                request_id=f"invalid_dir_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=[directory_path],
                pages=[],
                errors=["Invalid directory crawl request"],
                statistics=CrawlStatistics(),
            )

        # Execute directory crawling strategy
        return await self._directory_strategy.execute(request, progress_callback)

    async def crawl_repository(
        self,
        repo_url: str,
        clone_path: str | None = None,
        file_patterns: list[str] | None = None,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """
        Crawl a git repository using the optimized repository strategy.

        Args:
            repo_url: Git repository URL
            clone_path: Optional custom clone path
            file_patterns: File patterns to include
            progress_callback: Optional progress reporting callback

        Returns:
            CrawlResult with analyzed repository files and statistics
        """
        await self.initialize()

        self.logger.info(f"Starting repository crawl: {repo_url}")

        # Create repository request
        request = RepositoryRequest(
            repo_url=repo_url,
            clone_path=clone_path,
            file_patterns=file_patterns,
        )

        # Validate request
        if not await self._repository_strategy.validate_request(request):
            return CrawlResult(
                request_id=f"invalid_repo_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=[repo_url],
                pages=[],
                errors=["Invalid repository crawl request"],
                statistics=CrawlStatistics(),
            )

        # Execute repository crawling strategy
        return await self._repository_strategy.execute(request, progress_callback)

    async def scrape_single_page(
        self,
        url: str,
        extraction_strategy: str | None = None,
        wait_for: str | None = None,
        custom_config: dict[str, Any] | None = None,
        use_virtual_scroll: bool = False,
        virtual_scroll_config: dict[str, Any] | None = None,
    ) -> PageContent:
        """
        Scrape a single page using direct AsyncWebCrawler.

        Args:
            url: URL to scrape
            extraction_strategy: Extraction strategy to use
            wait_for: CSS selector or JS condition to wait for
            custom_config: Custom configuration options
            use_virtual_scroll: Whether to use virtual scrolling
            virtual_scroll_config: Virtual scroll configuration

        Returns:
            PageContent for the scraped page
        """
        await self.initialize()

        self.logger.debug(f"Scraping single page: {url}")

        from crawl4ai import AsyncWebCrawler, BrowserConfig  # type: ignore

        from ..types.crawl4ai_types import (
            DefaultMarkdownGeneratorImpl as DefaultMarkdownGenerator,
        )
        from ..types.crawl4ai_types import (
            PruningContentFilterImpl as PruningContentFilter,
        )

        # Create minimal browser config
        browser_config = BrowserConfig(
            headless=settings.crawl_headless,
            browser_type=settings.crawl_browser,
            light_mode=True,
            verbose=False,  # Suppress Crawl4AI output for MCP compatibility
            text_mode=getattr(settings, "crawl_block_images", False),
        )

        with suppress_stdout():
            browser = AsyncWebCrawler(config=browser_config)
            await browser.start()

        try:
            # Configure extraction strategy
            if extraction_strategy == "aggressive":
                # Aggressive pruning for cleaner content
                content_filter = None
                if PruningContentFilter is not None:
                    content_filter = PruningContentFilter(
                        threshold=0.6,  # Higher threshold for more aggressive pruning
                        threshold_type="dynamic",
                        min_word_threshold=10,  # Ignore shorter text blocks
                    )
            elif extraction_strategy == "minimal":
                # Minimal pruning to preserve more content
                content_filter = None
                if PruningContentFilter is not None:
                    content_filter = PruningContentFilter(
                        threshold=0.3,  # Lower threshold for less pruning
                        threshold_type="dynamic",
                        min_word_threshold=3,  # Keep even short text
                    )
            else:
                # FIXED: More conservative default strategy for better content extraction
                content_filter = None
                if PruningContentFilter is not None:
                    content_filter = PruningContentFilter(
                        threshold=0.25,  # REDUCED from 0.45 to 0.25 - keep 75% of content
                        threshold_type="dynamic",  # Dynamic scoring
                        min_word_threshold=3,  # REDUCED from 5 to 3 - keep shorter text blocks
                    )

            # Create markdown generator with content filter
            markdown_generator = None
            if DefaultMarkdownGenerator is not None:
                markdown_generator = DefaultMarkdownGenerator(
                    content_filter=content_filter
                )

            # Configure crawl parameters
            crawl_kwargs = {
                "url": url,
                "bypass_cache": not settings.crawl_enable_caching,
                "process_iframes": False,  # Disable for performance
                "remove_overlay_elements": settings.crawl_remove_overlays,
                "word_count_threshold": settings.crawl_min_words,
                # FIXED: Reduced excluded tags - only exclude script/style, keep nav/footer/header/aside
                "excluded_tags": [
                    "script",
                    "style",
                ],
                "exclude_external_links": True,
                "markdown_generator": markdown_generator,  # Enable fit markdown generation
            }

            # Apply virtual scrolling configuration if requested
            if use_virtual_scroll:
                crawl_kwargs["scroll_behavior"] = "smooth"
                crawl_kwargs["wait_for_network_idle"] = True

                if virtual_scroll_config:
                    # Apply custom virtual scroll settings
                    if "scroll_count" in virtual_scroll_config:
                        crawl_kwargs["page_timeout"] = (
                            virtual_scroll_config["scroll_count"] * 1000
                        )
                    if "scroll_delay" in virtual_scroll_config:
                        crawl_kwargs["delay_after_scroll"] = virtual_scroll_config[
                            "scroll_delay"
                        ]
                else:
                    # Use default virtual scroll settings from config
                    if (
                        hasattr(settings, "crawl_virtual_scroll")
                        and settings.crawl_virtual_scroll
                    ):
                        crawl_kwargs["page_timeout"] = (
                            settings.crawl_scroll_count * 1000
                        )
                        crawl_kwargs["delay_after_scroll"] = settings.crawl_scroll_delay

            # Apply wait_for condition if specified
            if wait_for:
                crawl_kwargs["wait_for"] = wait_for

            # Apply custom configuration overrides
            if custom_config:
                crawl_kwargs.update(custom_config)

            # Use Crawl4AI to scrape the page with fit markdown optimization
            with suppress_stdout():
                result = await browser.arun(**crawl_kwargs)

            if not result.success:
                raise Exception(f"Scraping failed: {result.error_message}")

            # Extract content using proper crawl4ai patterns
            # result.markdown is a MarkdownGenerationResult object with raw_markdown and fit_markdown attributes
            best_content = ""
            if result.markdown:
                try:
                    # Extract markdown content following crawl4ai best practices
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

            # Basic content validation
            if best_content and len(best_content.strip()) < 3:
                self.logger.debug(
                    f"Content too short for {url} (less than 3 chars), clearing"
                )
                best_content = ""

            # Simple fallback if markdown extraction failed
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

            # Create PageContent with validated content
            page_content = PageContent(
                url=url,
                title=result.metadata.get("title", ""),
                content=best_content,
                html=result.html,
                markdown=best_content,  # Use the validated content as markdown
                word_count=0,  # Will be calculated by validator from content
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
                    "extraction_strategy": extraction_strategy or "default",
                    "virtual_scroll_used": use_virtual_scroll,
                    "wait_for_condition": wait_for,
                    "status_code": result.status_code,
                    "response_headers": dict(result.response_headers or {}),
                },
                timestamp=datetime.utcnow(),
                # word_count will be calculated by the validator
            )

            return page_content

        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            raise
        finally:
            with suppress_stdout():
                await browser.close()

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of all crawler components."""
        await self.initialize()

        health_status: dict[str, Any] = {
            "crawler_service": "healthy",
            "components": {},
            "performance_optimizations": {
                "memory_threshold": f"{settings.crawl_memory_threshold}%",
                "direct_browser_usage": True,
                "light_mode_enabled": True,
                "streaming_enabled": settings.crawl_enable_streaming,
                "caching_enabled": settings.crawl_enable_caching,
            },
        }

        # Memory manager health
        if self._memory_manager:
            memory_stats = self._memory_manager.get_stats()
            health_status["components"]["memory_manager"] = {
                "status": "healthy",
                "stats": memory_stats,
            }

        return health_status

    async def _cleanup(self) -> None:
        """Service-specific cleanup."""
        self.logger.info("Cleaning up crawler service")

        # Cleanup memory manager
        if self._memory_manager:
            cleanup_memory_manager()

        self.logger.info("Crawler service cleanup completed")
