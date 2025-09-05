"""
Browser configuration factory for optimized high-performance web crawler.

This module provides factory methods for creating optimized browser configurations
for different crawling scenarios, focusing on performance and resource efficiency.
"""

from crawl4ai import BrowserConfig

from crawler_mcp.optimized_config import OptimizedConfig


class BrowserFactory:
    """Factory for creating optimized browser configurations"""

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize browser factory.

        Args:
            config: Optional optimized crawler configuration
        """
        self.config = config or OptimizedConfig()

    def create_performance_config(self) -> BrowserConfig:
        """
        Create high-performance browser configuration optimized for speed.

        This configuration is optimized for maximum crawling throughput with:
        - Headless mode for reduced overhead
        - Disabled GPU and graphics acceleration
        - Minimal browser features enabled
        - Optimized for text content extraction

        Returns:
            BrowserConfig optimized for performance
        """
        extra_args = [
            # Core performance optimizations
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            # Speed optimizations
            "--disable-extensions",
            "--disable-plugins",
            "--disable-default-apps",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            # Resource loading / stability optimizations
            "--aggressive-cache-discard",
            "--disable-hang-monitor",
            "--disable-prompt-on-repost",
            "--disable-sync",
            "--disable-translate",
            "--disable-ipc-flooding-protection",
        ]

        # Add content-specific optimizations based on config
        if self.config.text_mode:
            # Prefer blocking via crawler config; use blink setting as a hint only
            extra_args.append("--blink-settings=imagesEnabled=false")

        return BrowserConfig(
            headless=self.config.browser_headless,
            browser_type="chromium",  # Use Chromium for best performance
            extra_args=extra_args,
            viewport_width=800,  # Minimal viewport
            viewport_height=600,
            text_mode=self.config.text_mode,
            light_mode=True,  # Reduce browser features
            ignore_https_errors=True,  # Ignore SSL issues
            java_script_enabled=self.config.javascript_enabled,
            user_agent=self._get_realistic_user_agent(),
            storage_state=None,  # No persistent storage
        )

    def create_javascript_config(self) -> BrowserConfig:
        """
        Create browser configuration with JavaScript enabled for dynamic content.

        This configuration enables JavaScript while maintaining performance optimizations
        for sites that require client-side rendering.

        Returns:
            BrowserConfig with JavaScript enabled
        """
        base_config = self.create_performance_config()

        # Enable JavaScript and modify args
        base_config.java_script_enabled = True
        # Keep args lean; rely on run config for JS rendering controls

        return base_config

    def create_stealth_config(self) -> BrowserConfig:
        """
        Create browser configuration with anti-detection features.

        This configuration attempts to avoid detection as an automated browser
        while maintaining reasonable performance.

        Returns:
            BrowserConfig with stealth features
        """
        extra_args = [
            # Anti-detection
            "--disable-blink-features=AutomationControlled",
            "--exclude-switches=enable-automation",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            # Appear more like a real browser
            "--disable-plugins-discovery",
            "--start-maximized",
        ]

        return BrowserConfig(
            headless=False,  # Some sites detect headless mode
            browser_type="chromium",
            extra_args=extra_args,
            viewport_width=1920,  # Common desktop resolution
            viewport_height=1080,
            user_agent=self._get_realistic_user_agent(),
            text_mode=False,  # Load images to appear normal
            java_script_enabled=True,
            ignore_https_errors=True,
        )

    def create_minimal_config(self) -> BrowserConfig:
        """
        Create ultra-minimal browser configuration for maximum speed.

        This is the fastest possible configuration, suitable for simple HTML pages
        where maximum throughput is prioritized over compatibility.

        Returns:
            BrowserConfig with minimal features
        """
        return BrowserConfig(
            headless=True,
            browser_type="chromium",
            extra_args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-plugins",
                "--disable-extensions",
                "--virtual-time-budget=1000",  # Fast timeout
            ],
            viewport_width=400,  # Tiny viewport
            viewport_height=300,
            text_mode=True,
            java_script_enabled=False,
            user_agent=self._get_realistic_user_agent(),
        )

    def create_aggressive_config(self) -> BrowserConfig:
        """
        Create aggressive configuration for maximum concurrent performance.

        This configuration is optimized for high-concurrency scenarios where
        multiple browser instances will be running simultaneously.

        Returns:
            BrowserConfig optimized for concurrent usage
        """
        base_config = self.create_performance_config()

        # Aggressive mode focuses on reduced background work; avoid single-process constraints
        base_config.extra_args.extend(
            [
                "--disable-background-networking",
                "--disable-background-downloads",
                "--disable-features=TranslateUI",
                "--disable-component-extensions-with-background-pages",
            ]
        )

        return base_config

    def create_config_for_scenario(self, scenario: str) -> BrowserConfig:
        """
        Create browser configuration for specific crawling scenarios.

        Args:
            scenario: Crawling scenario ('performance', 'javascript', 'stealth',
                     'minimal', 'aggressive')

        Returns:
            BrowserConfig appropriate for the scenario

        Raises:
            ValueError: If scenario is not recognized
        """
        scenario_map = {
            "performance": self.create_performance_config,
            "javascript": self.create_javascript_config,
            "stealth": self.create_stealth_config,
            "minimal": self.create_minimal_config,
            "aggressive": self.create_aggressive_config,
        }

        if scenario not in scenario_map:
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Available scenarios: {list(scenario_map.keys())}"
            )

        return scenario_map[scenario]()

    def customize_config(
        self,
        base_config: BrowserConfig,
        custom_args: list[str] | None = None,
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
    ) -> BrowserConfig:
        """
        Customize an existing browser configuration.

        Args:
            base_config: Base configuration to customize
            custom_args: Additional browser arguments
            viewport: Custom viewport dimensions
            user_agent: Custom user agent string

        Returns:
            Customized BrowserConfig
        """
        # Create a copy to avoid modifying the original
        customized = BrowserConfig(
            headless=base_config.headless,
            browser_type=base_config.browser_type,
            extra_args=base_config.extra_args.copy() if base_config.extra_args else [],
            viewport_width=getattr(base_config, "viewport_width", 800),
            viewport_height=getattr(base_config, "viewport_height", 600),
            text_mode=base_config.text_mode,
            light_mode=getattr(base_config, "light_mode", True),
            java_script_enabled=base_config.java_script_enabled,
            user_agent=getattr(base_config, "user_agent", ""),
        )

        # Apply customizations
        if custom_args:
            customized.extra_args.extend(custom_args)

        if viewport:
            if "width" in viewport:
                customized.viewport_width = int(viewport["width"])
            if "height" in viewport:
                customized.viewport_height = int(viewport["height"])

        if user_agent:
            customized.user_agent = user_agent

        return customized

    def _get_realistic_user_agent(self) -> str:
        """
        Get a realistic user agent string for stealth mode.

        Returns:
            Realistic user agent string
        """
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    def get_recommended_config(self) -> BrowserConfig:
        """
        Get recommended configuration based on the current OptimizedConfig settings.

        Returns:
            BrowserConfig optimized for current settings
        """
        if self.config.stealth_mode:
            return self.create_stealth_config()
        elif self.config.use_aggressive_mode:
            return self.create_aggressive_config()
        elif self.config.javascript_enabled:
            return self.create_javascript_config()
        else:
            return self.create_performance_config()
