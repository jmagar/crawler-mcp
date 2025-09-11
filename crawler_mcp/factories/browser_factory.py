"""
Browser configuration factory for the crawler, derived from CrawlerSettings.

This avoids depending on OptimizedConfig and uses the unified settings.
"""

from __future__ import annotations

from typing import Any

from crawl4ai import BrowserConfig

from crawler_mcp.constants import (
    MINIMAL_VIEWPORT_HEIGHT,
    MINIMAL_VIEWPORT_WIDTH,
    BrowserMode,
)
from crawler_mcp.settings import CrawlerSettings


class BrowserFactory:
    """Factory for creating browser configurations using CrawlerSettings."""

    def __init__(
        self, settings: CrawlerSettings, overrides: dict[str, Any] | None = None
    ):
        self.settings = settings
        self.overrides = overrides or {}

    def _get(self, key: str, default: Any) -> Any:
        return self.overrides.get(key, getattr(self.settings, key, default))

    def create_config(self) -> BrowserConfig:
        mode = self._get("browser_mode", BrowserMode.HEADLESS)
        if isinstance(mode, str):
            # Normalize string values if any
            try:
                mode = BrowserMode(mode)
            except Exception:
                mode = BrowserMode.HEADLESS

        if mode == BrowserMode.TEXT:
            return self._create_text_config()
        if mode == BrowserMode.MINIMAL:
            return self._create_minimal_config()
        # FULL or HEADLESS fall back to full config, honoring headless flag
        return self._create_full_config()

    def get_recommended_config(self) -> BrowserConfig:
        return self.create_config()

    def _base_kwargs(self) -> dict[str, Any]:
        return {
            "browser_type": self._get("browser_type", "chromium"),
            "viewport_width": int(self._get("browser_width", MINIMAL_VIEWPORT_WIDTH)),
            "viewport_height": int(
                self._get("browser_height", MINIMAL_VIEWPORT_HEIGHT)
            ),
            "light_mode": True,
            "java_script_enabled": bool(self._get("browser_js_enabled", True)),
        }

    def _create_full_config(self) -> BrowserConfig:
        # Use headless by default for WSL2 compatibility
        headless = True  # Always headless for now
        return BrowserConfig(
            headless=headless,
            **self._base_kwargs(),
        )

    def _create_text_config(self) -> BrowserConfig:
        kwargs = self._base_kwargs()
        kwargs.update(
            {
                "text_mode": True,
                "java_script_enabled": False,
            }
        )
        return BrowserConfig(headless=True, **kwargs)

    def _create_minimal_config(self) -> BrowserConfig:
        kwargs = self._base_kwargs()
        kwargs.update(
            {
                "text_mode": True,
                "java_script_enabled": False,
                "viewport_width": MINIMAL_VIEWPORT_WIDTH,
                "viewport_height": MINIMAL_VIEWPORT_HEIGHT,
            }
        )
        return BrowserConfig(headless=True, **kwargs)
