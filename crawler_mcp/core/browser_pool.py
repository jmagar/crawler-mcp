"""
High-performance browser pool for concurrent operations.

Configurable for different hardware setups and environments.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

# Safe default browser arguments that work in all environments
SAFE_DEFAULT_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
]

# Hardware-specific profiles for optional optimization
HARDWARE_PROFILES = {
    "rtx4070_i7": [
        # RTX 4070 + i7-13700k optimized Chrome flags
        "--disable-gpu-sandbox",
        "--max_old_space_size=4096",  # 4GB per browser instance
        "--js-flags=--max-old-space-size=4096",
        "--renderer-process-limit=4",  # Limit renderer processes
        "--process-per-site",
        "--aggressive-cache-discard",
        "--memory-pressure-off",
        "--enable-gpu-rasterization",
        "--enable-zero-copy",
        "--enable-oop-rasterization",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-features=TranslateUI",
        "--no-zygote",  # Better for concurrent instances
        "--max-renderer-processes=4",
        "--enable-accelerated-2d-canvas",
        "--enable-gpu-compositing",
    ],
    "high_memory": [
        "--max_old_space_size=8192",  # 8GB per browser instance
        "--js-flags=--max-old-space-size=8192",
        "--renderer-process-limit=8",
        "--max-renderer-processes=8",
        "--aggressive-cache-discard",
        "--memory-pressure-off",
    ],
    "basic": [
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--renderer-process-limit=2",
        "--max-renderer-processes=2",
    ],
}


def _deduplicate_args(args: list[str]) -> list[str]:
    """Remove duplicate arguments, keeping the last occurrence."""
    seen = set()
    result = []
    for arg in reversed(args):
        # Extract the flag name (before '=' if present)
        flag_name = arg.split("=")[0]
        if flag_name not in seen:
            seen.add(flag_name)
            result.append(arg)
    return list(reversed(result))


class HighPerformanceBrowserPool:
    """Configurable high-performance browser pool for concurrent operations."""

    def __init__(self, pool_size: int = 8):
        self.pool_size = pool_size
        self.browsers: list[Any] = []  # Will be AsyncWebCrawler instances
        self.available_browsers: asyncio.Queue[Any] = asyncio.Queue(maxsize=pool_size)
        self.is_initialized = False
        self._init_lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def initialize(self) -> None:
        """Initialize browser pool with hardware-optimized config."""
        if self.is_initialized:
            return

        async with self._init_lock:
            # Re-check inside the lock to prevent race condition
            if self.is_initialized:
                return

            try:
                # Import here to avoid circular imports
                from crawl4ai import AsyncWebCrawler, BrowserConfig

                # Build browser arguments from configuration
                browser_args = SAFE_DEFAULT_ARGS.copy()

                # Add hardware profile if specified
                if settings.browser_hardware_profile:
                    profile_args = HARDWARE_PROFILES.get(
                        settings.browser_hardware_profile, []
                    )
                    browser_args.extend(profile_args)
                    logger.info(
                        f"Using hardware profile: {settings.browser_hardware_profile}"
                    )

                # Add user-specified extra arguments
                if settings.browser_extra_args:
                    browser_args.extend(settings.browser_extra_args)
                    logger.debug(
                        f"Added {len(settings.browser_extra_args)} custom browser args"
                    )

                # Deduplicate arguments (keeping last occurrence)
                browser_args = _deduplicate_args(browser_args)

                # Configuration-based browser setup - use consistent crawl settings
                browser_config = BrowserConfig(
                    headless=settings.crawl_headless,
                    browser_type=settings.crawl_browser,
                    verbose=False,  # Disable verbose output for production stability
                    extra_args=browser_args,
                )

                self.logger.info(
                    f"Initializing browser pool with {self.pool_size} browsers"
                )

                # Create browser instances
                for i in range(self.pool_size):
                    try:
                        browser = AsyncWebCrawler(config=browser_config)
                        await browser.__aenter__()
                        self.browsers.append(browser)
                        await self.available_browsers.put(browser)
                        self.logger.debug(
                            f"Initialized browser {i + 1}/{self.pool_size}"
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to initialize browser {i + 1}: {e}")
                        # Continue with fewer browsers rather than failing completely

                if not self.browsers:
                    raise RuntimeError("Failed to initialize any browsers")

                self.is_initialized = True
                self.logger.info(
                    f"Browser pool initialized successfully with {len(self.browsers)} browsers"
                )

            except Exception as e:
                self.logger.error(f"Failed to initialize browser pool: {e}")
                await self.cleanup()
                raise

    async def acquire(self) -> Any:
        """Get browser from pool."""
        if not self.is_initialized:
            await self.initialize()

        browser = await self.available_browsers.get()
        self.logger.debug("Browser acquired from pool")
        return browser

    async def release(self, browser: Any) -> None:
        """Return browser to pool."""
        try:
            await self.available_browsers.put(browser)
            self.logger.debug("Browser returned to pool")
        except Exception as e:
            self.logger.error(f"Failed to return browser to pool: {e}")

    @contextlib.asynccontextmanager
    async def lease(self) -> AsyncGenerator[Any, None]:
        """Context manager for safe browser acquisition and release."""
        browser = None
        try:
            browser = await self.acquire()
            self.logger.debug("Browser leased via context manager")
            yield browser
        finally:
            if browser is not None:
                await self.release(browser)
                self.logger.debug("Browser returned via context manager")

    async def cleanup(self) -> None:
        """Cleanup all browsers."""
        self.logger.info("Cleaning up browser pool")

        # Close all browsers
        for browser in self.browsers:
            try:
                await browser.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error closing browser: {e}")

        self.browsers.clear()

        # Clear the queue and mark tasks as done
        while not self.available_browsers.empty():
            try:
                self.available_browsers.get_nowait()
                # Mark the task as done to allow queue.join() to complete
                self.available_browsers.task_done()
            except asyncio.QueueEmpty:
                # Queue is empty, nothing more to drain
                break

        self.is_initialized = False
        self.logger.info("Browser pool cleanup completed")

    async def __aenter__(self) -> "HighPerformanceBrowserPool":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()

    @property
    def available_count(self) -> int:
        """Get number of available browsers in pool."""
        return self.available_browsers.qsize()

    @property
    def total_browsers(self) -> int:
        """Get total number of browsers in pool."""
        return len(self.browsers)


# Global browser pool instance
_browser_pool: HighPerformanceBrowserPool | None = None


async def get_browser_pool(pool_size: int = 8) -> HighPerformanceBrowserPool:
    """Get or create the global browser pool."""
    global _browser_pool

    if _browser_pool is None:
        # Try to get pool size from config, fallback to provided or default value
        try:
            from crawler_mcp.config import settings

            size = getattr(settings, "browser_pool_size", pool_size)
        except Exception:
            size = pool_size
        _browser_pool = HighPerformanceBrowserPool(pool_size=size)

    return _browser_pool


async def cleanup_browser_pool() -> None:
    """Cleanup the global browser pool."""
    global _browser_pool

    if _browser_pool is not None:
        await _browser_pool.cleanup()
        _browser_pool = None
