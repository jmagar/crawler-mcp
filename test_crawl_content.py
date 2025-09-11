#!/usr/bin/env python
"""Test what Crawl4AI returns for markdown content."""

import asyncio

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


async def test_crawl():
    """Test single page crawl to see what's in the markdown field."""

    # Create content filter and markdown generator
    content_filter = PruningContentFilter(threshold=0.45)
    markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)

    # Create config - similar to what's in strategy.py
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        word_count_threshold=10,
        excluded_tags=["script", "style", "nav", "footer"],
        markdown_generator=markdown_generator,
        stream=True,  # Test WITH streaming to see if this causes hash
    )

    print("Starting crawl...")
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun("https://gofastmcp.com", config=config)

        print(f"\nResult type: {type(result)}")
        print(f"Result.success: {result.success}")
        print(f"Result.url: {result.url}")

        # Check markdown field
        markdown = result.markdown
        print(f"\nMarkdown type: {type(markdown)}")
        print(
            f"Markdown attrs: {dir(markdown) if not isinstance(markdown, str) else 'string'}"
        )

        # Try to get content
        if hasattr(markdown, "fit_markdown"):
            content = markdown.fit_markdown
            print(f"\nfit_markdown type: {type(content)}")
            print(f"fit_markdown length: {len(str(content))}")
            print(f"fit_markdown preview: {str(content)[:200]}")
        elif hasattr(markdown, "raw_markdown"):
            content = markdown.raw_markdown
            print(f"\nraw_markdown type: {type(content)}")
            print(f"raw_markdown length: {len(str(content))}")
            print(f"raw_markdown preview: {str(content)[:200]}")
        elif isinstance(markdown, str):
            print("\nMarkdown is string")
            print(f"Length: {len(markdown)}")
            print(f"Preview: {markdown[:200]}")
            # Check if it's a hash
            if len(markdown) < 100 and all(c in "0123456789abcdef" for c in markdown):
                print(f"WARNING: Markdown appears to be a hash: {markdown}")
        else:
            print("\nUnknown markdown format")
            print(f"String repr: {str(markdown)[:200]}")

        # Also check cleaned_html
        if hasattr(result, "cleaned_html"):
            print(f"\ncleaned_html length: {len(result.cleaned_html)}")
            print(f"cleaned_html preview: {result.cleaned_html[:200]}")


if __name__ == "__main__":
    asyncio.run(test_crawl())
