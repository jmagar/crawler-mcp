#!/usr/bin/env python
"""Test what's in the markdown field during streaming crawl."""

import asyncio

from crawl4ai import AsyncWebCrawler, BFSDeepCrawlStrategy, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


async def test_streaming():
    """Test what markdown content looks like in streaming mode."""

    # Create content filter and markdown generator
    content_filter = PruningContentFilter(threshold=0.45)
    markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)

    # Create deep crawl strategy
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=1,
        max_pages=3,
        include_external=False,
        url_scorer=None,
        score_threshold=0.0,
    )

    # Create config with streaming
    config = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl_strategy,
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        word_count_threshold=10,
        excluded_tags=["script", "style", "nav", "footer"],
        markdown_generator=markdown_generator,
        stream=True,
    )

    print("Testing streaming crawl markdown content...")
    async with AsyncWebCrawler(verbose=False) as crawler:
        page_num = 0
        async for result in await crawler.arun("https://gofastmcp.com", config=config):
            page_num += 1
            print(f"\n{'=' * 60}")
            print(f"Page {page_num}: {result.url}")
            print(f"Success: {result.success}")

            if result.success and hasattr(result, "markdown"):
                markdown = result.markdown

                # Check fit_markdown
                fit_content = ""
                if hasattr(markdown, "fit_markdown"):
                    fit_content = markdown.fit_markdown or ""
                    print(f"fit_markdown length: {len(fit_content)}")
                    print(
                        f"fit_markdown preview: {fit_content[:100] if fit_content else 'EMPTY'}"
                    )

                # Check raw_markdown
                raw_content = ""
                if hasattr(markdown, "raw_markdown"):
                    raw_content = markdown.raw_markdown or ""
                    print(f"raw_markdown length: {len(raw_content)}")
                    print(
                        f"raw_markdown preview: {raw_content[:100] if raw_content else 'EMPTY'}"
                    )

                # What's being used in strategy.py
                markdown_content = fit_content if fit_content else raw_content
                print(
                    f"\nUsed in logging: '{markdown_content[:50]}...' (length: {len(markdown_content)})"
                )

            if page_num >= 3:
                break


if __name__ == "__main__":
    asyncio.run(test_streaming())
