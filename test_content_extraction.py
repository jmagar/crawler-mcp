#!/usr/bin/env python3
"""
Quick test to check if content is in the response HTML field.
"""

import asyncio

from crawler_mcp.crawlers.optimized import OptimizedConfig, OptimizedCrawlerStrategy


async def test_content_extraction():
    """Test content extraction from response"""

    config = OptimizedConfig(
        max_concurrent_crawls=1,
        max_urls_to_discover=1,
        check_robots_txt=False,
        use_rate_limiting=False,
    )

    strategy = OptimizedCrawlerStrategy(config)

    try:
        await strategy.start()

        # Use crawl_single_url which returns our internal format
        result = await strategy.crawl_single_url("https://gofastmcp.com", max_urls=1)

        print(f"Success: {result['success']}")
        print(f"Content length: {len(result['content'])}")
        print(f"HTML length: {len(result['html'])}")
        print(f"Metadata: {result['metadata']}")

        if result["content"]:
            print(f"Content preview: {result['content'][:200]}...")
        elif result["html"]:
            print(f"HTML preview: {result['html'][:200]}...")
        else:
            print("NO CONTENT OR HTML!")

    finally:
        await strategy.close()


if __name__ == "__main__":
    asyncio.run(test_content_extraction())
