"""
Content extraction factory for optimized high-performance web crawler.

This module provides factory methods for creating content extraction configurations
that prevent hash placeholders and ensure high-quality markdown content extraction.
"""

from crawl4ai import CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from ..config import OptimizedConfig


class ContentExtractorFactory:
    """Factory for content extraction configurations that prevent hash placeholders"""

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize content extractor factory.

        Args:
            config: Optional optimized crawler configuration
        """
        self.config = config or OptimizedConfig()

    def create_markdown_generator(
        self,
        content_threshold: float | None = None,
        min_word_threshold: int | None = None,
    ) -> DefaultMarkdownGenerator:
        """
        Create optimized markdown generator with content filtering.

        This generator uses dynamic content filtering to prevent hash placeholders
        and extract high-quality markdown content.

        Args:
            content_threshold: Override default content relevance threshold
            min_word_threshold: Override default minimum word threshold

        Returns:
            DefaultMarkdownGenerator configured for optimal content extraction
        """
        # Use provided values or fall back to config defaults
        # (Currently not used because content_filter is disabled by default to avoid placeholders.)

        # TEMP FIX: Disable content filtering to prevent hash placeholders
        # The PruningContentFilter is causing arun_many() to return hash placeholders
        content_filter = None

        # Configure markdown generation options
        markdown_options = {
            "skip_internal_links": False,  # Keep internal links for crawling
            "skip_external_links": True,  # Remove external links for focus
            "include_images": self.config.extract_images,
            "extract_tables": True,  # Preserve table data
            "heading_style": "atx",  # Use # heading style
            "code_block_style": "fenced",  # Use ``` code blocks
            "emphasis_style": "underscore",  # Use _ for emphasis
            "link_style": "inline",  # Use [text](url) format
        }

        return DefaultMarkdownGenerator(
            content_filter=content_filter, options=markdown_options
        )

    def create_aggressive_markdown_generator(self) -> DefaultMarkdownGenerator:
        """
        Create markdown generator optimized for maximum content extraction.

        This generator uses more aggressive filtering to extract content from
        challenging pages while still preventing hash placeholders.

        Returns:
            DefaultMarkdownGenerator with aggressive content extraction
        """
        # More aggressive content filter
        content_filter = PruningContentFilter(
            threshold=0.3,  # Lower threshold for more content
            threshold_type="dynamic",
            min_word_threshold=3,  # Very lenient word threshold
            min_content_length=20,  # Lower minimum length
            remove_empty_elements=True,
            exclude_tags_by_depth=False,  # Don't exclude by depth
            max_depth=15,  # Deeper processing
            preserve_content_structure=False,  # Flatten structure for more content
        )

        return DefaultMarkdownGenerator(
            content_filter=content_filter,
            options={
                "skip_internal_links": False,
                "skip_external_links": False,  # Keep all links
                "include_images": True,
                "extract_tables": True,
                "preserve_formatting": True,
            },
        )

    def create_minimal_markdown_generator(self) -> DefaultMarkdownGenerator:
        """
        Create lightweight markdown generator for maximum speed.

        This generator prioritizes speed over comprehensive content extraction,
        suitable for scenarios where basic text content is sufficient.

        Returns:
            DefaultMarkdownGenerator optimized for speed
        """
        # Minimal content filter
        content_filter = PruningContentFilter(
            threshold=0.6,  # Higher threshold for speed
            threshold_type="fixed",  # Fixed threshold for consistency
            min_word_threshold=10,
            min_content_length=100,
            remove_empty_elements=True,
            exclude_tags_by_depth=True,
            max_depth=5,  # Shallow processing
        )

        return DefaultMarkdownGenerator(
            content_filter=content_filter,
            options={
                "skip_internal_links": True,  # Skip links for speed
                "skip_external_links": True,
                "include_images": False,
                "extract_tables": False,  # Skip tables for speed
                "preserve_formatting": False,
            },
        )

    def create_crawler_config(
        self,
        markdown_gen: DefaultMarkdownGenerator | None = None,
        excluded_tags: list[str] | None = None,
        check_robots: bool | None = None,
    ) -> CrawlerRunConfig:
        """
        Create crawler configuration with content extraction settings.

        Args:
            markdown_gen: Optional markdown generator (creates default if None)
            excluded_tags: Optional list of HTML tags to exclude
            check_robots: Whether to check robots.txt (defaults to config setting)

        Returns:
            CrawlerRunConfig optimized for content extraction
        """
        # Use provided generator or create default
        if markdown_gen is None:
            markdown_gen = self.create_markdown_generator()

        # Use provided tags or config defaults
        if excluded_tags is None:
            excluded_tags = self.config.excluded_tags.copy()

        # Use provided robots setting or config default
        if check_robots is None:
            check_robots = self.config.check_robots_txt

        rc = CrawlerRunConfig(
            # Content extraction
            markdown_generator=markdown_gen,
            # Tag filtering to prevent noise
            excluded_tags=excluded_tags,
            # Link handling
            exclude_external_links=True,  # Focus on domain content
            exclude_social_media_links=True,  # Remove social media noise
            # Performance and caching
            cache_mode=CacheMode.ENABLED
            if self.config.enable_cache
            else CacheMode.BYPASS,
            check_robots_txt=check_robots,
            # Content quality thresholds
            word_count_threshold=self.config.min_word_count,
            # Page processing optimizations
            remove_overlay_elements=True,  # Remove popups and overlays
            process_iframes=False,  # Skip iframes for performance
            # Timing optimizations
            page_timeout=self.config.page_timeout,
            delay_before_return_html=0.5,  # Brief delay for content loading
            # Additional optimizations
            only_text=self.config.text_mode,  # Text-only mode for speed
            verbose=False,  # Disable verbose output for performance
        )

        # Align run config with global JS setting for dynamic docs sites
        try:
            if hasattr(rc, "enable_javascript"):
                rc.enable_javascript = bool(self.config.javascript_enabled)
            if self.config.javascript_enabled:
                # Allow some time for hydration and DOM stabilization
                if hasattr(rc, "wait_for_js_rendering"):
                    rc.wait_for_js_rendering = True
                try:
                    rc.delay_before_return_html = max(
                        1.0, float(rc.delay_before_return_html)
                    )  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

        return rc

    def create_quality_focused_config(self) -> CrawlerRunConfig:
        """
        Create configuration focused on content quality over speed.

        This configuration prioritizes extracting high-quality, comprehensive
        content at the expense of some performance.

        Returns:
            CrawlerRunConfig optimized for content quality
        """
        # Use aggressive markdown generator for quality
        markdown_gen = self.create_aggressive_markdown_generator()

        # Minimal tag exclusions for comprehensive content
        quality_excluded_tags = ["script", "style", "noscript"]

        return CrawlerRunConfig(
            markdown_generator=markdown_gen,
            excluded_tags=quality_excluded_tags,
            exclude_external_links=False,  # Keep all links
            cache_mode=CacheMode.ENABLED,
            check_robots_txt=False,  # Always ignore robots.txt
            word_count_threshold=20,  # Lower threshold for quality
            remove_overlay_elements=True,
            process_iframes=True,  # Process iframes for completeness
            page_timeout=60000,  # Longer timeout for quality
            delay_before_return_html=2.0,  # More time for content loading
            enable_javascript=True,  # Enable JS for dynamic content
            wait_for_js_rendering=True,
            verbose=False,
        )

    def create_speed_focused_config(self) -> CrawlerRunConfig:
        """
        Create configuration focused on maximum crawling speed.

        This configuration prioritizes speed over comprehensive content extraction,
        suitable for scenarios where basic content is sufficient.

        Returns:
            CrawlerRunConfig optimized for speed
        """
        # Use minimal markdown generator for speed
        markdown_gen = self.create_minimal_markdown_generator()

        # Aggressive tag exclusions for speed
        speed_excluded_tags = [
            "nav",
            "header",
            "footer",
            "aside",
            "script",
            "style",
            "noscript",
            "iframe",
            "form",
            "button",
            "input",
        ]

        return CrawlerRunConfig(
            markdown_generator=markdown_gen,
            excluded_tags=speed_excluded_tags,
            exclude_external_links=True,
            exclude_social_media_links=True,
            cache_mode=CacheMode.ENABLED,
            check_robots_txt=False,
            word_count_threshold=self.config.min_word_count,
            remove_overlay_elements=True,
            process_iframes=False,
            page_timeout=15000,  # Shorter timeout
            delay_before_return_html=0.1,  # Minimal delay
            enable_javascript=False,  # Disable JS for speed
            only_text=True,  # Text-only for maximum speed
            verbose=False,
        )

    def create_config_for_content_type(self, content_type: str) -> CrawlerRunConfig:
        """
        Create configuration optimized for specific content types.

        Args:
            content_type: Type of content ('article', 'documentation', 'product',
                         'news', 'blog', 'general')

        Returns:
            CrawlerRunConfig optimized for the content type

        Raises:
            ValueError: If content_type is not recognized
        """
        content_configs = {
            "article": self._create_article_config,
            "documentation": self._create_documentation_config,
            "product": self._create_product_config,
            "news": self._create_news_config,
            "blog": self._create_blog_config,
            "general": self.create_crawler_config,
        }

        if content_type not in content_configs:
            raise ValueError(
                f"Unknown content type: {content_type}. "
                f"Available types: {list(content_configs.keys())}"
            )

        return content_configs[content_type]()

    def _create_article_config(self) -> CrawlerRunConfig:
        """Create config optimized for article content"""
        markdown_gen = self.create_markdown_generator(
            content_threshold=0.4,  # Medium threshold
            min_word_threshold=5,
        )

        article_excluded_tags = [
            "nav",
            "header",
            "footer",
            "aside",
            "script",
            "style",
            "advertisement",
            "ad",
            "sidebar",
        ]

        return self.create_crawler_config(
            markdown_gen=markdown_gen, excluded_tags=article_excluded_tags
        )

    def _create_documentation_config(self) -> CrawlerRunConfig:
        """Create config optimized for documentation content"""
        # Documentation benefits from comprehensive extraction
        markdown_gen = self.create_aggressive_markdown_generator()

        doc_excluded_tags = ["nav", "footer", "script", "style"]  # Minimal exclusions

        config = self.create_crawler_config(
            markdown_gen=markdown_gen, excluded_tags=doc_excluded_tags
        )
        config.word_count_threshold = 10  # Lower threshold for code snippets
        return config

    def _create_product_config(self) -> CrawlerRunConfig:
        """Create config optimized for product pages"""
        markdown_gen = self.create_markdown_generator(
            content_threshold=0.45, min_word_threshold=3
        )

        product_excluded_tags = [
            "nav",
            "header",
            "footer",
            "script",
            "style",
            "review-widget",
            "recommendation",
        ]

        return self.create_crawler_config(
            markdown_gen=markdown_gen, excluded_tags=product_excluded_tags
        )

    def _create_news_config(self) -> CrawlerRunConfig:
        """Create config optimized for news content"""
        markdown_gen = self.create_markdown_generator(
            content_threshold=0.5,  # Higher threshold for quality
            min_word_threshold=8,
        )

        news_excluded_tags = [
            "nav",
            "header",
            "footer",
            "aside",
            "script",
            "style",
            "advertisement",
            "related-articles",
            "trending",
        ]

        return self.create_crawler_config(
            markdown_gen=markdown_gen, excluded_tags=news_excluded_tags
        )

    def _create_blog_config(self) -> CrawlerRunConfig:
        """Create config optimized for blog content"""
        markdown_gen = self.create_markdown_generator(
            content_threshold=0.42, min_word_threshold=6
        )

        blog_excluded_tags = [
            "nav",
            "header",
            "footer",
            "sidebar",
            "script",
            "style",
            "comment-section",
            "author-bio",
            "social-share",
        ]

        return self.create_crawler_config(
            markdown_gen=markdown_gen, excluded_tags=blog_excluded_tags
        )
