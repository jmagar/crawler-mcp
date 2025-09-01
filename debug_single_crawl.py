#!/usr/bin/env python3
"""
Debug script to test single URL crawl and inspect actual content.
"""

import asyncio
import logging

from crawler_mcp.crawlers.optimized import OptimizedConfig, OptimizedCrawlerStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def debug_single_crawl():
    """Debug single URL crawl to see actual content"""

    print("🔍 Debug Single URL Crawl")
    print("=" * 40)

    config = OptimizedConfig(
        max_concurrent_crawls=1,
        max_urls_to_discover=1,
        check_robots_txt=False,
        use_rate_limiting=False,
        use_aggressive_mode=False,  # Turn off aggressive mode
    )

    strategy = OptimizedCrawlerStrategy(config)

    test_url = "https://gofastmcp.com"

    try:
        await strategy.start()

        # First, let's try a direct crawl4ai test
        print("\n🧪 Testing direct crawl4ai...")
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=test_url)

            print(f"Direct crawl success: {result.success}")
            print(f"Status code: {result.status_code}")

            if result.markdown:
                content = str(result.markdown)
                print(f"Content length: {len(content)}")
                print(f"First 500 chars: {content[:500]}...")
                print(f"Word count: {len(content.split())}")
            else:
                print("❌ No markdown content!")

        print("\n" + "=" * 40)
        print("🔧 Testing our optimized strategy...")

        # Now test our strategy with debugging
        response = await strategy.crawl(test_url, max_urls=1)

        print(f"Strategy success: {getattr(response, 'success', 'N/A')}")
        print(f"Response type: {type(response)}")
        print(f"HTML length: {len(response.html) if response.html else 0}")

        # Try to access metadata if it exists
        if hasattr(response, "__dict__"):
            print(f"Response fields: {list(response.__dict__.keys())}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await strategy.close()


if __name__ == "__main__":
    asyncio.run(debug_single_crawl())
