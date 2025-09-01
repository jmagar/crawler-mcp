"""
Optimized high-performance web crawler package.

This package provides a modular, high-performance web crawler implementation
that uses Crawl4AI patterns for maximum throughput and content quality.

Key Features:
- URL discovery from sitemaps without initial crawling
- Parallel processing with arun_many() for high concurrency
- Memory-adaptive dispatching for resource optimization
- Content extraction with hash placeholder prevention
- Comprehensive performance monitoring and hooks
- No rate limiting for maximum speed (as requested)
- Ignores robots.txt for aggressive crawling (as requested)

Usage Example:
    from crawler_mcp.crawlers.optimized import OptimizedCrawlerStrategy, OptimizedConfig

    # Configure for maximum performance
    config = OptimizedConfig(
        max_concurrent_crawls=16,
        check_robots_txt=False,
        use_rate_limiting=False
    )

    # Create and use strategy
    strategy = OptimizedCrawlerStrategy(config)

    async def main():
        await strategy.start()
        try:
            response = await strategy.crawl("https://example.com")
            print(f"Crawled {response.metadata['pages_crawled']} pages")
        finally:
            await strategy.close()
"""

# Core strategy and configuration
# Component factories
from .config import OptimizedConfig
from .core.parallel_engine import CrawlStats, ParallelEngine
from .core.strategy import OptimizedCrawlerStrategy
from .factories.browser_factory import BrowserFactory
from .factories.content_extractor import ContentExtractorFactory
from .factories.dispatcher_factory import DispatcherFactory
from .processing.result_converter import ResultConverter

# Core engines and utilities
from .processing.url_discovery import URLDiscovery
from .utils.monitoring import CrawlMetrics, PerformanceMonitor

# Package metadata
__version__ = "1.0.0"
__author__ = "Optimized Crawler Team"
__description__ = "High-performance web crawler with Crawl4AI integration"

# Main exports - what users should typically import
__all__ = [
    "BrowserFactory",
    "ContentExtractorFactory",
    "CrawlMetrics",
    "CrawlStats",
    "DispatcherFactory",
    "OptimizedConfig",
    "OptimizedCrawlerStrategy",
    "ParallelEngine",
    "PerformanceMonitor",
    "ResultConverter",
    "URLDiscovery",
]

# Version info
VERSION_INFO = (1, 0, 0)
