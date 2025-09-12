"""
Result conversion utilities for optimized high-performance web crawler.

This module provides utilities for converting between Crawl4AI models and our
existing MCP models, enabling seamless integration with the existing system.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from crawl4ai.models import CrawlResult as Crawl4AIResult

from crawler_mcp.constants import IMAGE_EXTENSIONS

# Import our existing models
from crawler_mcp.models.crawl import (
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from crawler_mcp.settings import CrawlerSettings


class ResultConverter:
    """Converts between Crawl4AI models and our MCP models"""

    def __init__(
        self, settings: CrawlerSettings, overrides: dict[str, Any] | None = None
    ):
        """Initialize result converter.

        Args:
            settings: Global settings instance
            overrides: Optional runtime configuration overrides
        """
        self.settings = settings
        self.overrides = overrides or {}
        self.logger = logging.getLogger(__name__)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value from overrides or settings."""
        return self.overrides.get(key, getattr(self.settings, key, default))

    def crawl4ai_to_page_content(
        self, result: Crawl4AIResult, prefer_fit_markdown: bool = True
    ) -> PageContent:
        """
        Convert Crawl4AI result to our PageContent model.

        Args:
            result: Crawl4AI crawl result
            prefer_fit_markdown: Whether to prefer fit_markdown over raw_markdown

        Returns:
            PageContent object compatible with our existing system
        """
        try:
            # Extract markdown content using best practices
            content = self._extract_markdown_content(result, prefer_fit_markdown)

            # Extract title from metadata or HTML
            title = self._extract_title(result)

            # Extract and process links
            links = (
                self._extract_links(result)
                if self.get_config_value("extract_links", False)
                else []
            )

            # Extract and process images
            images = (
                self._extract_images(result)
                if self.get_config_value("extract_images", False)
                else []
            )

            # Calculate word count
            word_count = len(content.split()) if content else 0

            # Extract metadata
            metadata = self._extract_metadata(result)

            return PageContent(
                url=result.url,
                title=title,
                content=content,
                html=getattr(result, "html", ""),
                markdown=content,  # Use same content for markdown field
                word_count=word_count,
                links=links,
                images=images,
                links_count=len(links),
                images_count=len(images),
                metadata=metadata,
                timestamp=datetime.now(UTC),
            )

        except Exception as e:
            self.logger.error(
                "Failed to convert Crawl4AI result for %s: %s",
                result.url,
                e,
                exc_info=True,
            )

            # Return minimal PageContent on error
            return PageContent(
                url=getattr(result, "url", ""),
                title="",
                content="",
                html="",
                markdown="",
                word_count=0,
                links=[],
                images=[],
                links_count=0,
                images_count=0,
                metadata={"conversion_error": str(e)},
                timestamp=datetime.now(UTC),
            )

    def _extract_markdown_content(
        self, result: Crawl4AIResult, prefer_fit_markdown: bool = True
    ) -> str:
        """
        Extract markdown content from Crawl4AI result.

        Args:
            result: Crawl4AI result
            prefer_fit_markdown: Whether to prefer fit_markdown

        Returns:
            Extracted markdown content
        """
        if not hasattr(result, "markdown") or not result.markdown:
            return ""

        try:
            markdown_obj = result.markdown
            self.logger.debug(
                "Markdown object type: %s, hasattr fit_markdown: %s, hasattr raw_markdown: %s",
                type(markdown_obj),
                hasattr(markdown_obj, "fit_markdown"),
                hasattr(markdown_obj, "raw_markdown"),
            )

            # Try fit_markdown first if preferred
            if prefer_fit_markdown and hasattr(markdown_obj, "fit_markdown"):
                fit_content = markdown_obj.fit_markdown
                self.logger.debug(
                    "fit_markdown content type: %s, length: %d",
                    type(fit_content),
                    len(str(fit_content)) if fit_content else 0,
                )
                if fit_content and str(fit_content).strip():
                    return str(fit_content).strip()

            # Try raw_markdown as fallback
            if hasattr(markdown_obj, "raw_markdown"):
                raw_content = markdown_obj.raw_markdown
                self.logger.debug(
                    "raw_markdown content type: %s, length: %d",
                    type(raw_content),
                    len(str(raw_content)) if raw_content else 0,
                )
                if raw_content and str(raw_content).strip():
                    return str(raw_content).strip()

            # Direct string conversion as last resort
            if isinstance(markdown_obj, str):
                self.logger.debug(
                    "markdown_obj is string, length: %d", len(markdown_obj)
                )
                return markdown_obj.strip()

            # Try string conversion of the object
            content = str(markdown_obj).strip()
            # Determine if truncated
            truncated = len(content) > 100
            preview = content[:100]
            suffix = "…" if truncated else ""
            self.logger.debug(
                "String conversion preview='%s%s' length=%d",
                preview,
                suffix,
                len(content),
            )
            return content if content != "None" else ""

        except Exception as e:
            self.logger.debug("Failed to extract markdown from %s: %s", result.url, e)
            return ""

    def _extract_title(self, result: Crawl4AIResult) -> str:
        """Extract title from Crawl4AI result"""
        try:
            # Try metadata first
            if (
                hasattr(result, "metadata")
                and result.metadata
                and isinstance(result.metadata, dict)
            ):
                title = result.metadata.get("title", "")
                if title:
                    return str(title).strip()

            # Try other title sources
            if hasattr(result, "title") and result.title:
                return str(result.title).strip()

            return ""

        except Exception:
            return ""

    def _extract_links(self, result: Crawl4AIResult) -> list[str]:
        """Extract links from Crawl4AI result"""
        try:
            links = []

            if hasattr(result, "links") and result.links:
                if isinstance(result.links, dict):
                    # Internal links are always included
                    internal = result.links.get("internal", [])
                    if isinstance(internal, list):
                        links.extend([self._normalize_link(link) for link in internal])

                    # External links included when extract_links is True
                    if self.get_config_value("extract_links", False):
                        external = result.links.get("external", [])
                        if isinstance(external, list):
                            links.extend(
                                [self._normalize_link(link) for link in external]
                            )

                elif isinstance(result.links, list):
                    links.extend([self._normalize_link(link) for link in result.links])

            # Filter and validate links
            return [link for link in links if self._is_valid_link(link)]

        except Exception as e:
            self.logger.debug("Failed to extract links from %s: %s", result.url, e)
            return []

    def _extract_images(self, result: Crawl4AIResult) -> list[str]:
        """Extract images from Crawl4AI result"""
        try:
            images = []

            if (
                hasattr(result, "media")
                and result.media
                and isinstance(result.media, dict)
            ):
                image_list = result.media.get("images", [])
                if isinstance(image_list, list):
                    for img in image_list:
                        src = img.get("src", "") if isinstance(img, dict) else str(img)

                        if src:
                            images.append(self._normalize_image_url(src))

            # Filter and validate images
            return [img for img in images if self._is_valid_image_url(img)]

        except Exception as e:
            self.logger.debug("Failed to extract images from %s: %s", result.url, e)
            return []

    def _extract_metadata(self, result: Crawl4AIResult) -> dict[str, Any]:
        """Extract metadata from Crawl4AI result"""
        metadata = {}

        try:
            # Add basic crawl information
            metadata.update(
                {
                    "status_code": getattr(result, "status_code", None),
                    "success": getattr(result, "success", False),
                    "extraction_method": "crawl4ai_optimized",
                    "crawl_timestamp": datetime.now(UTC).isoformat(),
                }
            )

            # Add timing information if available
            if hasattr(result, "crawl_time"):
                metadata["crawl_time"] = result.crawl_time

            # Add original metadata if present
            if (
                hasattr(result, "metadata")
                and result.metadata
                and isinstance(result.metadata, dict)
            ):
                # Add original metadata with prefix to avoid conflicts
                for key, value in result.metadata.items():
                    if key not in metadata:  # Don't override our metadata
                        metadata[f"original_{key}"] = value

            # Add response headers if available
            if hasattr(result, "response_headers") and result.response_headers:
                metadata["response_headers"] = dict(result.response_headers)

            # Add error information if present
            if hasattr(result, "error") and result.error:
                metadata["error"] = str(result.error)

            return metadata

        except Exception as e:
            self.logger.debug("Failed to extract metadata from %s: %s", result.url, e)
            return {"metadata_error": str(e)}

    def _normalize_link(self, link: str | dict[str, Any]) -> str:
        """Normalize link to string format"""
        if isinstance(link, dict):
            return link.get("href", link.get("url", ""))
        return str(link)

    def _normalize_image_url(self, img: str | dict[str, Any]) -> str:
        """Normalize image to URL string"""
        if isinstance(img, dict):
            return img.get("src", img.get("url", ""))
        return str(img)

    def _is_valid_link(self, link: str) -> bool:
        """Validate link URL"""
        if not link or not isinstance(link, str):
            return False

        # Basic URL validation
        try:
            parsed = urlparse(link)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def _is_valid_image_url(self, img_url: str) -> bool:
        """Validate image URL with strict validation"""
        if not img_url or not isinstance(img_url, str) or len(img_url.strip()) == 0:
            return False

        img_url = img_url.strip()

        # Reject obviously invalid URLs
        if len(img_url) > 2048:  # URL too long
            return False

        # Check for data URLs first (more specific validation)
        if img_url.startswith("data:"):
            return img_url.startswith("data:image/") and "," in img_url

        try:
            parsed = urlparse(img_url)

            # Only allow http, https schemes
            if parsed.scheme not in ("http", "https"):
                return False

            # Require valid netloc for http/https URLs
            if not parsed.netloc or len(parsed.netloc) < 3:
                return False

            path = parsed.path.lower()

            # Check for valid image extensions (primary validation)
            if any(path.endswith(ext) for ext in IMAGE_EXTENSIONS):
                return True

            # More restrictive query parameter validation
            query_params = parsed.query.lower()
            image_indicators = [
                "format=jpg",
                "format=jpeg",
                "format=png",
                "format=webp",
                "format=gif",
                "type=image/",
                "mime=image/",
                "filetype=image",
            ]
            # Return True if any image indicators found, False otherwise
            return any(indicator in query_params for indicator in image_indicators)

        except Exception:
            return False

    def batch_to_crawl_result(
        self,
        crawl4ai_results: list[Crawl4AIResult],
        original_urls: list[str],
        start_time: float,
        request_id: str | None = None,
        errors: list[str] | None = None,
    ) -> CrawlResult:
        """
        Convert batch of Crawl4AI results to our CrawlResult model.

        Args:
            crawl4ai_results: List of Crawl4AI results
            original_urls: Original URLs that were requested
            start_time: Start time of the crawl session
            request_id: Optional request identifier
            errors: Optional list of errors that occurred

        Returns:
            CrawlResult compatible with our existing system
        """
        try:
            # Convert individual results to PageContent
            pages = []
            for result in crawl4ai_results:
                page_content = self.crawl4ai_to_page_content(result)
                pages.append(page_content)

            # Calculate statistics
            end_time = time.time()
            duration = end_time - start_time
            statistics = self._calculate_statistics(
                pages, original_urls, duration, errors or []
            )

            # Generate request ID if not provided
            if request_id is None:
                request_id = f"optimized_{int(time.time())}"

            return CrawlResult(
                request_id=request_id,
                status=CrawlStatus.COMPLETED,
                urls=original_urls,
                pages=pages,
                errors=errors or [],
                statistics=statistics,
            )

        except Exception as e:
            self.logger.error("Failed to convert batch results: %s", e, exc_info=True)

            # Return failed result
            return CrawlResult(
                request_id=request_id or f"failed_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=original_urls,
                pages=[],
                errors=[str(e)],
                statistics=CrawlStatistics(),
            )

    def _calculate_statistics(
        self,
        pages: list[PageContent],
        original_urls: list[str],
        duration: float,
        errors: list[str],
    ) -> CrawlStatistics:
        """Calculate crawl statistics"""
        try:
            # Basic counts
            total_requested = len(original_urls)
            total_crawled = len(pages)
            total_failed = len(errors)

            # Content statistics
            total_bytes = sum(len(page.content) for page in pages)
            # Derive links count from links list for compatibility with different models
            total_links = 0
            for page in pages:
                try:
                    links = getattr(page, "links", [])
                    total_links += len(links) if links else 0
                except Exception:
                    continue

            # Domain analysis
            unique_domains = set()
            for page in pages:
                try:
                    domain = urlparse(page.url).netloc
                    if domain:
                        unique_domains.add(domain)
                except Exception:
                    continue

            # Performance metrics
            pages_per_second = total_crawled / duration if duration > 0 else 0
            average_page_size = total_bytes / total_crawled if total_crawled > 0 else 0

            return CrawlStatistics(
                total_pages_requested=total_requested,
                total_pages_crawled=total_crawled,
                total_pages_failed=total_failed,
                unique_domains=len(unique_domains),
                total_links_discovered=total_links,
                total_bytes_downloaded=total_bytes,
                crawl_duration_seconds=duration,
                pages_per_second=pages_per_second,
                average_page_size=average_page_size,
            )

        except Exception as e:
            self.logger.error("Failed to calculate statistics: %s", e, exc_info=True)
            return CrawlStatistics()

    def create_minimal_page_content(
        self, url: str, error: str | None = None
    ) -> PageContent:
        """
        Create minimal PageContent for failed URLs.

        Args:
            url: URL that failed
            error: Optional error message

        Returns:
            Minimal PageContent object
        """
        metadata = {"crawl_failed": True}
        if error:
            metadata["error"] = error

        return PageContent(
            url=url,
            title="",
            content="",
            html="",
            markdown="",
            word_count=0,
            links=[],
            images=[],
            links_count=0,
            images_count=0,
            metadata=metadata,
            timestamp=datetime.now(UTC),
        )

    def validate_conversion(
        self, crawl4ai_result: Crawl4AIResult, page_content: PageContent
    ) -> bool:
        """
        Validate that conversion was successful.

        Args:
            crawl4ai_result: Original Crawl4AI result
            page_content: Converted PageContent

        Returns:
            True if conversion appears successful
        """
        try:
            # Check basic fields
            if page_content.url != crawl4ai_result.url:
                return False

            # Check content exists if original was successful
            if crawl4ai_result.success and not page_content.content:
                return False

            # Check word count is reasonable
            if page_content.content:
                return page_content.word_count != 0
            return True

        except Exception:
            return False
