"""
URL discovery using Crawl4AI's built-in BFSDeepCrawlStrategy.

This module provides an adapter that uses Crawl4AI's documented deep crawl
strategy instead of custom sitemap/robots.txt parsing.
"""

import logging

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

from crawler_mcp.optimized_config import OptimizedConfig


class URLDiscoveryAdapter:
    """Adapter for Crawl4AI's BFS discovery strategy."""

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize URL discovery adapter.

        Args:
            config: Optional optimized crawler configuration
        """
        self.config = config or OptimizedConfig()
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    async def discover_all(
        self, start_url: str, max_urls: int | None = None
    ) -> list[str]:
        """
        Discover URLs using BFSDeepCrawlStrategy.

        Args:
            start_url: Starting URL to discover from
            max_urls: Maximum URLs to return (defaults to config value)

        Returns:
            List of discovered URLs
        """
        max_urls = max_urls or self.config.max_urls_to_discover

        try:
            # Create URL pattern filter to stay within domain
            from urllib.parse import urlparse

            domain = urlparse(start_url).netloc
            url_filter = URLPatternFilter(
                patterns=[f"https://{domain}/*", f"http://{domain}/*"], action="include"
            )

            # Create relevance scorer with keywords
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
                threshold=self.config.url_score_threshold,
            )

            # Create BFS strategy with appropriate depth
            strategy = BFSDeepCrawlStrategy(
                max_depth=3,  # Reasonable depth for most sites
                filter_chain=url_filter,
                scorer=scorer,
                max_urls=max_urls,
            )

            # Configure crawler for discovery
            config = CrawlerRunConfig(
                deep_crawl_strategy=strategy,
                page_timeout=self.config.page_timeout,
                word_count_threshold=self.config.min_word_count,
                excluded_tags=["script", "style", "noscript"],
            )

            # Run discovery crawl
            self.logger.info(
                f"Starting BFS discovery from {start_url} (max: {max_urls})"
            )

            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=start_url, config=config)

                if not result.success:
                    self.logger.warning(
                        f"Discovery failed for {start_url}: {result.error_message}"
                    )
                    return [start_url]  # Fallback to original URL

            # Extract discovered URLs from strategy
            discovered_urls = getattr(strategy, "discovered_urls", [])

            if not discovered_urls:
                self.logger.info("No URLs discovered via BFS, using original URL")
                return [start_url]

            # Limit to max_urls and ensure start_url is included
            final_urls = list(dict.fromkeys([start_url, *discovered_urls]))[:max_urls]

            self.logger.info(
                f"BFS discovery found {len(final_urls)} URLs from {start_url}"
            )
            return final_urls

        except Exception as e:
            self.logger.error(f"BFS discovery failed: {e}")
            # Fallback to original URL
            return [start_url]
