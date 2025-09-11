#!/usr/bin/env python
"""Test deep crawl with BFSDeepCrawlStrategy."""

import asyncio

from crawl4ai import AsyncWebCrawler, BFSDeepCrawlStrategy, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


async def test_deep_crawl():
    """Test deep crawl to see what's in the markdown field."""

    # Create content filter and markdown generator
    content_filter = PruningContentFilter(threshold=0.45)
    markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)

    # Create deep crawl strategy
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=1,
        max_pages=3,
        include_external=False,
        url_scorer=None,  # Disable scoring
        score_threshold=0.0,
    )

    # Create config with deep crawl strategy
    config = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl_strategy,
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        word_count_threshold=10,
        excluded_tags=["script", "style", "nav", "footer"],
        markdown_generator=markdown_generator,
        stream=True,
    )

    print("Starting deep crawl with stream=True...")
    async with AsyncWebCrawler(verbose=True) as crawler:
        # For deep crawl, use arun with the strategy
        async for result in await crawler.arun("https://gofastmcp.com", config=config):
            print(f"\n{'=' * 60}")
            print(f"Result for: {result.url}")
            print(f"Success: {result.success}")

            # Check markdown field
            markdown = result.markdown
            print(f"Markdown type: {type(markdown)}")

            # Try to get content
            if hasattr(markdown, "fit_markdown"):
                content = markdown.fit_markdown
            elif hasattr(markdown, "raw_markdown"):
                content = markdown.raw_markdown
            elif isinstance(markdown, str):
                content = markdown
            else:
                content = str(markdown)

            print(f"Content length: {len(str(content))}")
            print(f"Content preview: {str(content)[:100]}")

            # Check if it's a hash
            if len(str(content)) < 100 and all(
                c in "0123456789abcdef" for c in str(content)
            ):
                print(f"WARNING: Content appears to be a hash: {content}")

            if hasattr(result, "cleaned_html"):
                print(
                    f"cleaned_html length: {len(result.cleaned_html) if result.cleaned_html else 0}"
                )


if __name__ == "__main__":
    asyncio.run(test_deep_crawl())
