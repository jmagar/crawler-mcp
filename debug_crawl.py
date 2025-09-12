#!/usr/bin/env python3
"""
Debug script to test Crawl4AI deep crawling behavior for gofastmcp.com
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain


async def debug_crawl():
    """Test crawl with minimal configuration to isolate issues."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    url = "https://gofastmcp.com"
    logger.info(f"🚀 Starting debug crawl of {url}")

    # Simple domain filter (needs both should_include and apply methods for Crawl4AI compatibility)
    class SimpleDomainFilter:
        def __init__(self, domain: str):
            self.domain = domain.lower()

        def should_include(self, url: str) -> bool:
            try:
                from urllib.parse import urlparse

                if url.startswith("/"):
                    return True
                parsed = urlparse(url)
                if not parsed.scheme and not parsed.netloc:
                    return True
                return parsed.netloc.lower() == self.domain
            except Exception:
                return False

        def apply(self, url: str) -> bool:
            """Alias for should_include to match Crawl4AI filter interface"""
            return self.should_include(url)

    domain_filter = SimpleDomainFilter("gofastmcp.com")

    # Create minimal deep crawl strategy
    strategy = BFSDeepCrawlStrategy(
        max_depth=2,  # Reduced for testing
        max_pages=5,  # Reduced for testing
        include_external=False,
        filter_chain=FilterChain([domain_filter]),
        url_scorer=None,  # Disable scoring
        score_threshold=0.0,  # Disable threshold
    )

    # Minimal browser config
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        ignore_https_errors=True,
        verbose=True,
    )

    # Minimal crawler config (using only valid CrawlerRunConfig parameters)
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        stream=True,
        cache_mode=CacheMode.BYPASS,  # Avoid cache issues
        page_timeout=60000,  # 60 seconds
        word_count_threshold=1,  # Very permissive
        wait_until="domcontentloaded",
        delay_before_return_html=1.0,  # Reduced delay
        semaphore_count=3,  # Reduced concurrency
        mean_delay=0.5,
        max_range=1.0,
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            logger.info("🔍 Starting streaming crawl...")

            page_count = 0
            successful_pages = 0
            failed_pages = 0

            async for result in await crawler.arun(url, config=config):
                page_count += 1

                logger.info(f"📄 Processing page {page_count}: {result.url}")
                logger.info(f"   Success: {result.success}")
                logger.info(f"   Status: {getattr(result, 'status_code', 'N/A')}")
                logger.info(
                    f"   Redirected: {getattr(result, 'redirected_url', 'None')}"
                )
                logger.info(f"   HTML length: {len(getattr(result, 'html', '') or '')}")
                logger.info(f"   Error: {getattr(result, 'error_message', 'None')}")

                if result.success:
                    successful_pages += 1
                    # Check for links in the result
                    links = getattr(result, "links", {})
                    internal_links = links.get("internal", []) if links else []
                    external_links = links.get("external", []) if links else []
                    logger.info(f"   Internal links found: {len(internal_links)}")
                    logger.info(f"   External links found: {len(external_links)}")

                    if internal_links:
                        logger.info("   📍 First 3 internal links:")
                        for link in internal_links[:3]:
                            logger.info(f"      → {link}")
                else:
                    failed_pages += 1

                logger.info("-" * 80)

                # Stop after reasonable number for debugging
                if page_count >= 10:
                    logger.info("⏹️  Stopping after 10 pages for debugging")
                    break

            logger.info(
                f"✅ Crawl complete: {successful_pages} successful, {failed_pages} failed out of {page_count} total"
            )

            if successful_pages == 0:
                logger.error(
                    "🚨 NO SUCCESSFUL PAGES - this explains why deep crawling stops!"
                )
            elif successful_pages == 1 and page_count == 1:
                logger.error(
                    "🚨 ONLY FIRST PAGE SUCCESSFUL - check if links are being extracted and filtered properly!"
                )

    except Exception as e:
        logger.error(f"❌ Crawl failed with exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_crawl())
