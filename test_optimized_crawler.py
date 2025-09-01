#!/usr/bin/env python3
"""
Test script for the new OptimizedCrawlerStrategy implementation.
"""

import asyncio
import logging
import time

from crawler_mcp.crawlers.optimized import OptimizedConfig, OptimizedCrawlerStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_optimized_crawler():
    """Test the optimized crawler on gofastmcp.com"""

    print("🚀 Testing OptimizedCrawlerStrategy")
    print("=" * 50)

    # Create optimized configuration
    config = OptimizedConfig(
        max_concurrent_crawls=16,
        max_urls_to_discover=1000,  # Test with up to 1000 URLs
        check_robots_txt=False,
        use_rate_limiting=False,
        use_aggressive_mode=True,
    )

    print(f"Configuration: {config}")
    print()

    # Create strategy
    strategy = OptimizedCrawlerStrategy(config)

    # Test URL
    test_url = "https://gofastmcp.com"

    try:
        # Start the strategy
        print("Starting crawler strategy...")
        start_time = time.time()
        await strategy.start()

        print(f"Strategy started in {time.time() - start_time:.2f}s")
        print()

        # Execute crawl
        print(f"Starting optimized crawl of: {test_url}")
        print("Target: Up to 1000 URLs with 16 concurrent crawlers")
        print("⏱️  Crawl timer started...")
        crawl_start = time.time()

        # Test multi-URL crawling for performance - use main crawl method
        async_response = await strategy.crawl(test_url, max_urls=1000)

        crawl_duration = time.time() - crawl_start
        print(f"⏱️  CRAWL COMPLETED in {crawl_duration:.2f} seconds")

        # Convert AsyncCrawlResponse to our format for easier handling
        headers = async_response.response_headers or {}
        success = headers.get("X-Crawl-Success", "False").lower() == "true"

        # Extract content from HTML if it contains extracted content marker
        content = ""
        if async_response.html and "<!-- EXTRACTED_CONTENT -->" in async_response.html:
            start_marker = "<!-- EXTRACTED_CONTENT -->\n"
            end_marker = "\n<!-- END_CONTENT -->"
            start_idx = async_response.html.find(start_marker)
            end_idx = async_response.html.find(end_marker)
            if start_idx != -1 and end_idx != -1:
                content = async_response.html[start_idx + len(start_marker) : end_idx]

        response = {
            "success": success,
            "url": headers.get("X-Crawl-URL", test_url),
            "content": content,
            "html": async_response.html or "",
            "metadata": {
                "pages_crawled": int(headers.get("X-Pages-Crawled", "0")),
                "urls_discovered": int(headers.get("X-URLs-Discovered", "0")),
                "pages_per_second": float(headers.get("X-Pages-Per-Second", "0")),
                "extraction_method": headers.get("X-Extraction-Method", "unknown"),
                "content_length": int(headers.get("X-Content-Length", "0")),
                "error": headers.get("X-Crawl-Error", ""),
            },
            "links": {"internal": [], "external": []},
            "duration": 0,  # Will be set below
        }

        print()
        print("=" * 50)
        print("🎯 CRAWL RESULTS")
        print("=" * 50)

        # Print results - response is now a dict from crawl_single_url
        print(f"✅ Success: {response.get('success', False)}")
        print(f"⏱️  Total crawl duration: {crawl_duration:.2f} seconds")

        # Handle metadata from our internal format
        metadata = response.get("metadata", {})
        pages_crawled = metadata.get("pages_crawled", 0)
        urls_discovered = metadata.get("urls_discovered", 0)

        print(f"📄 Pages crawled: {pages_crawled}")
        print(f"📊 URLs discovered: {urls_discovered}")

        # Calculate actual pages per second from our timer
        actual_pages_per_sec = (
            pages_crawled / crawl_duration if crawl_duration > 0 else 0
        )
        print(f"⚡ Actual pages/second: {actual_pages_per_sec:.2f}")
        print(f"🎯 Extraction method: {metadata.get('extraction_method', 'unknown')}")

        # Performance summary
        if pages_crawled > 0:
            print(
                f"📈 Performance: {pages_crawled} pages in {crawl_duration:.1f}s = {actual_pages_per_sec:.2f} pages/sec"
            )

        # Content quality check
        content = response.get("content", "") or ""
        content_length = len(content)
        print(f"📝 Content length: {content_length:,} characters")

        if content_length > 0:
            word_count = len(content.split())
            print(f"🔤 Word count: {word_count:,} words")

            # Check for hash placeholders
            if word_count <= 10:
                print(
                    "⚠️  WARNING: Very low word count - possible hash placeholder issue"
                )
            else:
                print("✅ Content looks good - no hash placeholder detected")

        # Sample content preview
        if content:
            preview = content[:500]  # Show more content
            print(f"📖 Content preview: {preview}...")

            # Show first few URLs from links if available
            links = response.get("links", {})
            if isinstance(links, dict) and "internal" in links:
                internal_links = links["internal"][:5]  # First 5 links
                print(f"🔗 Sample links found: {internal_links}")
        else:
            print("⚠️  NO CONTENT EXTRACTED!")

        # Performance report
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE REPORT")
        print("=" * 50)

        try:
            perf_report = strategy.get_performance_report()
            summary = perf_report.get("summary", {})

            print(f"Success rate: {summary.get('success_rate', 0):.1%}")
            print(f"Total duration: {summary.get('total_duration', 0):.2f}s")
            print(f"Pages per second: {summary.get('pages_per_second', 0):.2f}")

            content_analysis = perf_report.get("content_analysis", {})
            print(
                f"Total content: {content_analysis.get('total_content_bytes', 0):,} bytes"
            )
            print(
                f"Average page size: {content_analysis.get('average_content_length', 0):.0f} bytes"
            )

            # Show recommendations if any
            recommendations = perf_report.get("recommendations", [])
            if recommendations:
                print("\n🔧 Recommendations:")
                for rec in recommendations:
                    print(f"  • {rec}")

        except Exception as e:
            print(f"⚠️  Could not get performance report: {e}")

        # DEBUG: Let's see what's actually in the response
        print("\n🔍 DEBUG INFO:")
        print(f"Response type: {type(response)}")
        print(f"Response keys: {list(response.keys())}")
        html_len = len(response.get("html", ""))
        print(f"HTML length: {html_len}")
        print(f"Full metadata: {response.get('metadata', {})}")

    except Exception as e:
        print(f"❌ Crawl failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        print("\n" + "=" * 50)
        print("🧹 CLEANUP")
        print("=" * 50)

        try:
            await strategy.close()
            print("✅ Strategy closed successfully")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")

        total_time = time.time() - start_time
        print(f"⏱️  Total test time: {total_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(test_optimized_crawler())
