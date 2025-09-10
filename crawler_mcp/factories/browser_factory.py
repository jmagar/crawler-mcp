"""
Browser configuration factory for optimized high-performance web crawler.

This module provides factory methods for creating browser configurations
using Crawl4AI's documented browser modes.
"""

from crawl4ai import BrowserConfig

from crawler_mcp.optimized_config import OptimizedConfig


class BrowserFactory:
    """Factory for creating optimized browser configurations using documented Crawl4AI APIs"""

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

    def create_documentation_config(self) -> BrowserConfig:
        """Create specialized config for documentation sites."""
        return BrowserConfig(
            headless=True,
            text_mode=True,
            browser_type="chromium",
            viewport_width=1280,
            viewport_height=720,
            light_mode=True,
            java_script_enabled=False,  # Docs usually don't need JS
            extra_args=[
                *self.config.browser_extra_args,
                "--disable-images",
                "--disable-javascript",
                "--disable-plugins",
            ],
        )

    def create_api_config(self) -> BrowserConfig:
        """Create specialized config for API documentation."""
        return BrowserConfig(
            headless=True,
            browser_type="chromium",
            viewport_width=1280,
            viewport_height=720,
            light_mode=True,
            java_script_enabled=True,  # API docs might need JS for examples
            extra_args=self.config.browser_extra_args,
        )

    def get_config_for_url(self, url: str) -> BrowserConfig:
        """Get optimized browser config based on URL pattern.

        Args:
            url: URL to analyze for optimal configuration

        Returns:
            BrowserConfig optimized for the detected content type
        """
        if not self.config.enable_url_based_optimization:
            return self.create_config()

        # Check for documentation patterns
        for pattern in self.config.documentation_patterns:
            if self._matches_pattern(url, pattern):
                return self.create_documentation_config()

        # Check for API documentation
        if "/api/" in url.lower() or "/api-" in url.lower():
            return self.create_api_config()

        # Default to configured mode
        return self.create_config()

    def _matches_pattern(self, url: str, pattern: str) -> bool:
        """Check if URL matches a glob-style pattern."""
        import fnmatch

        return fnmatch.fnmatch(url.lower(), pattern.lower())

    def _create_full_config(self) -> BrowserConfig:
        """Create full-featured browser configuration with performance optimizations."""
        return BrowserConfig(
            headless=self.config.browser_headless,
            browser_type="chromium",
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height,
            light_mode=self.config.enable_light_mode,
            java_script_enabled=self.config.enable_javascript,
            extra_args=self.config.browser_extra_args,
        )

    def _create_text_config(self) -> BrowserConfig:
        """Create optimized text-only browser configuration."""
        return BrowserConfig(
            headless=True,
            text_mode=True,
            browser_type="chromium",
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height,
            light_mode=True,  # Always enable for text mode
            java_script_enabled=False,  # Disable JS for text mode
            extra_args=[
                *self.config.browser_extra_args,
                "--disable-images",
                "--disable-javascript",
                "--disable-plugins",
                "--disable-extensions",
            ],
        )

    def _create_minimal_config(self) -> BrowserConfig:
        """Create minimal browser configuration with aggressive resource blocking."""
        return BrowserConfig(
            headless=True,
            text_mode=True,
            browser_type="chromium",
            viewport_width=800,  # Smaller viewport for minimal mode
            viewport_height=600,
            light_mode=True,
            java_script_enabled=False,
            extra_args=[
                *self.config.browser_extra_args,
                "--disable-images",
                "--disable-javascript",
                "--disable-plugins",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-background-timer-throttling",
                "--disable-client-side-phishing-detection",
                "--disable-default-apps",
                "--disable-hang-monitor",
                "--disable-popup-blocking",
                "--disable-prompt-on-repost",
                "--disable-sync",
                "--disable-translate",
                "--disable-web-security",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )
