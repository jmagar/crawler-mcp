"""
Browser configuration factory for optimized high-performance web crawler.

This module provides factory methods for creating browser configurations
using Crawl4AI's documented browser modes.
"""

from crawl4ai import BrowserConfig

from crawler_mcp.optimized_config import OptimizedConfig


class BrowserFactory:
    """Factory for creating browser configurations using documented Crawl4AI APIs"""

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize browser factory.

        Args:
            config: Optional optimized crawler configuration
        """
        self.config = config or OptimizedConfig()

    def create_config(self) -> BrowserConfig:
        """
        Create browser configuration based on configured browser_mode.

        Returns:
            BrowserConfig appropriate for the configured mode
        """
        if self.config.browser_mode == "text":
            return self._create_text_config()
        elif self.config.browser_mode == "minimal":
            return self._create_minimal_config()
        else:  # full
            return self._create_full_config()

    def get_recommended_config(self) -> BrowserConfig:
        """
        Get recommended browser configuration (alias for create_config).

        Returns:
            BrowserConfig appropriate for the configured mode
        """
        return self.create_config()

    def _create_full_config(self) -> BrowserConfig:
        """Create full-featured browser configuration."""
        return BrowserConfig(
            headless=self.config.browser_headless,
            browser_type="chromium",
        )

    def _create_text_config(self) -> BrowserConfig:
        """Create text-only browser configuration."""
        return BrowserConfig(
            headless=True,
            text_mode=True,
            browser_type="chromium",
        )

    def _create_minimal_config(self) -> BrowserConfig:
        """Create minimal browser configuration with aggressive resource blocking."""
        return BrowserConfig(
            headless=True,
            text_mode=True,
            disable_images=True,
            browser_type="chromium",
        )
