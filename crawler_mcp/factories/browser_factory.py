"""
Browser configuration factory for the crawler, derived from CrawlerSettings.

This avoids depending on OptimizedConfig and uses the unified settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crawl4ai import BrowserConfig
else:
    from crawl4ai import BrowserConfig as _RuntimeBrowserConfig

    BrowserConfig = _RuntimeBrowserConfig  # type: ignore[assignment]

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
            "light_mode": bool(self._get("light_mode", True)),
            "java_script_enabled": bool(self._get("browser_js_enabled", True)),
            "extra_args": list(self._get("extra_args", [])),
        }

    def _get_headless_mode(self) -> bool:
        """Determine headless mode based on browser_mode setting."""
        mode = self._get("browser_mode", BrowserMode.HEADLESS)
        if isinstance(mode, str):
            try:
                mode = BrowserMode(mode)
            except Exception:
                mode = BrowserMode.HEADLESS
        # FULL mode is not headless, all others are headless
        return mode is not BrowserMode.FULL

    def _create_full_config(self) -> BrowserConfig:
        # Get the browser mode from settings/overrides
        mode = self._get("browser_mode", BrowserMode.HEADLESS)
        if isinstance(mode, str):
            try:
                mode = BrowserMode(mode)
            except Exception:
                mode = BrowserMode.HEADLESS

        # Set headless to False when mode is FULL, True otherwise
        headless = False if mode == BrowserMode.FULL else True
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
        headless = self._get_headless_mode()
        return BrowserConfig(headless=headless, **kwargs)

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
        headless = self._get_headless_mode()
        return BrowserConfig(headless=headless, **kwargs)
