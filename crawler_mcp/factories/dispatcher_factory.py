"""
Memory-adaptive dispatcher factory for optimized high-performance web crawler.

This module provides factory methods for creating MemoryAdaptiveDispatcher instances
optimized for different crawling scenarios and hardware configurations.
"""

import inspect
import logging

from crawl4ai import MemoryAdaptiveDispatcher

try:  # Optional monitor integration
    from crawl4ai import CrawlerMonitor, DisplayMode
except ImportError:  # pragma: no cover
    CrawlerMonitor = None  # type: ignore
    DisplayMode = None  # type: ignore

from crawler_mcp.optimized_config import OptimizedConfig


class DispatcherFactory:
    """Factory for creating optimized MemoryAdaptiveDispatcher instances"""

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize dispatcher factory.

        Args:
            config: Optional optimized crawler configuration
        """
        self.config = config or OptimizedConfig()
        self.logger = logging.getLogger(__name__)

    def create_performance_dispatcher(
        self,
        max_concurrent: int | None = None,
        memory_threshold: float | None = None,
    ) -> MemoryAdaptiveDispatcher:
        """
        Create dispatcher optimized for high performance.

        Args:
            max_concurrent: Maximum concurrent crawls (defaults to config)
            memory_threshold: Memory threshold percentage (defaults to config)

        Returns:
            MemoryAdaptiveDispatcher optimized for performance
        """
        # Use provided values or config defaults
        concurrent = max_concurrent or self.config.max_concurrent_crawls
        threshold = memory_threshold or self.config.memory_threshold

        kwargs = {}
        if self._supports_monitor_kw():
            kwargs["monitor"] = self._maybe_build_monitor()
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=threshold,
            max_session_permit=concurrent,
            check_interval=0.5,
            **kwargs,
        )

    def create_aggressive_dispatcher(
        self, max_concurrent: int | None = None
    ) -> MemoryAdaptiveDispatcher:
        """
        Create dispatcher with aggressive memory usage for maximum performance.

        Args:
            max_concurrent: Maximum concurrent crawls (defaults to config * 1.5)

        Returns:
            MemoryAdaptiveDispatcher with aggressive settings
        """
        # Use aggressive defaults if not provided
        concurrent = max_concurrent or (self.config.max_concurrent_crawls * 1.5)

        kwargs = {}
        if self._supports_monitor_kw():
            kwargs["monitor"] = self._maybe_build_monitor()
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=85.0,  # Higher memory usage tolerance
            max_session_permit=int(concurrent),
            check_interval=0.25,  # Very frequent checks
            recovery_threshold_percent=80.0,  # Later recovery
            **kwargs,
        )

    def create_conservative_dispatcher(
        self, max_concurrent: int | None = None
    ) -> MemoryAdaptiveDispatcher:
        """
        Create conservative dispatcher for stable performance.

        Args:
            max_concurrent: Maximum concurrent crawls (defaults to config * 0.75)

        Returns:
            MemoryAdaptiveDispatcher with conservative settings
        """
        concurrent = max_concurrent or max(
            1, int(self.config.max_concurrent_crawls * 0.75)
        )

        kwargs = {}
        if self._supports_monitor_kw():
            kwargs["monitor"] = self._maybe_build_monitor()
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=60.0,  # Lower memory threshold
            max_session_permit=concurrent,
            check_interval=1.0,  # Less frequent checks
            **kwargs,
        )

    def create_memory_efficient_dispatcher(
        self, max_concurrent: int | None = None
    ) -> MemoryAdaptiveDispatcher:
        """
        Create dispatcher optimized for memory efficiency.

        Args:
            max_concurrent: Maximum concurrent crawls (defaults to config // 2)

        Returns:
            MemoryAdaptiveDispatcher optimized for low memory usage
        """
        concurrent = max_concurrent or max(1, self.config.max_concurrent_crawls // 2)

        kwargs = {}
        if self._supports_monitor_kw():
            kwargs["monitor"] = self._maybe_build_monitor()
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=50.0,  # Very conservative memory usage
            max_session_permit=concurrent,
            check_interval=2.0,  # Infrequent checks to reduce overhead
            **kwargs,
        )

    def create_large_scale_dispatcher(
        self, max_concurrent: int | None = None
    ) -> MemoryAdaptiveDispatcher:
        """
        Create dispatcher for large-scale crawling operations.

        Args:
            max_concurrent: Maximum concurrent crawls (defaults to config * 2)

        Returns:
            MemoryAdaptiveDispatcher for large-scale operations
        """
        concurrent = max_concurrent or (self.config.max_concurrent_crawls * 2)

        kwargs = {}
        if self._supports_monitor_kw():
            kwargs["monitor"] = self._maybe_build_monitor()
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=self.config.memory_threshold,
            max_session_permit=int(concurrent),
            check_interval=0.5,
            **kwargs,
        )

    def create_batch_dispatcher(
        self, batch_size: int, memory_threshold: float | None = None
    ) -> MemoryAdaptiveDispatcher:
        """
        Create dispatcher optimized for batch processing.

        Args:
            batch_size: Size of batches to process
            memory_threshold: Memory threshold percentage (defaults to config)

        Returns:
            MemoryAdaptiveDispatcher optimized for batch operations
        """
        threshold = memory_threshold or self.config.memory_threshold

        kwargs = {}
        if self._supports_monitor_kw():
            kwargs["monitor"] = self._maybe_build_monitor()
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=threshold,
            max_session_permit=batch_size,
            check_interval=1.0,
            **kwargs,
        )

    def create_custom_dispatcher(
        self, memory_threshold: float, max_concurrent: int, check_interval: float = 1.0
    ) -> MemoryAdaptiveDispatcher:
        """
        Create dispatcher with custom settings.

        Args:
            memory_threshold: Memory threshold percentage
            max_concurrent: Maximum concurrent crawls
            check_interval: Memory check interval in seconds

        Returns:
            MemoryAdaptiveDispatcher with custom settings
        """
        kwargs = {}
        if self._supports_monitor_kw():
            kwargs["monitor"] = self._maybe_build_monitor()
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=memory_threshold,
            max_session_permit=max_concurrent,
            check_interval=check_interval,
            **kwargs,
        )

    def get_recommended_dispatcher(self) -> MemoryAdaptiveDispatcher:
        """
        Get the recommended dispatcher based on configuration.

        Returns:
            MemoryAdaptiveDispatcher with recommended settings
        """
        if self.config.use_aggressive_mode:
            return self.create_aggressive_dispatcher()
        elif self.config.memory_threshold <= 60.0:
            return self.create_conservative_dispatcher()
        else:
            return self.create_performance_dispatcher()

    def create_testing_dispatcher(self) -> MemoryAdaptiveDispatcher:
        """
        Create dispatcher suitable for testing and development.

        Returns:
            MemoryAdaptiveDispatcher with testing-friendly settings
        """
        kwargs = {}
        if self._supports_monitor_kw():
            kwargs["monitor"] = self._maybe_build_monitor()
        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=75.0,
            max_session_permit=4,  # Small number for testing
            check_interval=1.0,
            **kwargs,
        )

    def _supports_monitor_kw(self) -> bool:
        """Check if MemoryAdaptiveDispatcher supports the monitor keyword."""
        try:
            return "monitor" in inspect.signature(MemoryAdaptiveDispatcher).parameters
        except Exception:
            return False

    def _maybe_build_monitor(self) -> "CrawlerMonitor | None":
        """Create a CrawlerMonitor if enabled and available."""
        if not getattr(self.config, "enable_crawler_monitor", False):
            return None
        if CrawlerMonitor is None or DisplayMode is None:
            self.logger.debug("CrawlerMonitor not available in this environment")
            return None
        try:
            # Use default constructor, let CrawlerMonitor handle its own defaults
            return CrawlerMonitor()
        except Exception as e:
            self.logger.debug("Failed to build CrawlerMonitor: %s", e)
            return None
