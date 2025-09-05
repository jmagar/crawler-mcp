"""
Result conversion utilities for optimized high-performance web crawler.

This module provides utilities for converting between Crawl4AI models and our
existing MCP models, enabling seamless integration with the existing system.
"""

import logging
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from crawl4ai.models import CrawlResult as Crawl4AIResult

from crawler_mcp.crawl_core.parallel_engine import CrawlStats

# Import our existing models
from crawler_mcp.models.crawl import (
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
    PageContent,
)
from crawler_mcp.optimized_config import OptimizedConfig


class ResultConverter:
    """Converts between Crawl4AI models and our MCP models"""

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize result converter.

        Args:
            config: Optional optimized crawler configuration
        """
        self.config = config or OptimizedConfig()
        self.logger = logging.getLogger(__name__)

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
            links = self._extract_links(result) if self.config.extract_links else []

            # Extract and process images
            images = self._extract_images(result) if self.config.extract_images else []

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
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(
                f"Failed to convert Crawl4AI result for {result.url}: {e}"
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
                timestamp=datetime.now(),
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

            # Try fit_markdown first if preferred
            if prefer_fit_markdown and hasattr(markdown_obj, "fit_markdown"):
                fit_content = markdown_obj.fit_markdown
                if fit_content and str(fit_content).strip():
                    return str(fit_content).strip()

            # Try raw_markdown as fallback
            if hasattr(markdown_obj, "raw_markdown"):
                raw_content = markdown_obj.raw_markdown
                if raw_content and str(raw_content).strip():
                    return str(raw_content).strip()

            # Direct string conversion as last resort
            if isinstance(markdown_obj, str):
                return markdown_obj.strip()

            # Try string conversion of the object
            content = str(markdown_obj).strip()
            return content if content != "None" else ""

        except Exception as e:
            self.logger.debug(f"Failed to extract markdown from {result.url}: {e}")
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
                    if self.config.extract_links:
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
            self.logger.debug(f"Failed to extract links from {result.url}: {e}")
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
            self.logger.debug(f"Failed to extract images from {result.url}: {e}")
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
                    "crawl_timestamp": datetime.now().isoformat(),
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
            self.logger.debug(f"Failed to extract metadata from {result.url}: {e}")
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
        """Validate image URL"""
        if not img_url or not isinstance(img_url, str):
            return False

        # Check for common image extensions
        img_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp"}
        try:
            parsed = urlparse(img_url)
            path = parsed.path.lower()
            return (
                any(path.endswith(ext) for ext in img_extensions)
                or "image" in img_url.lower()
            )
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
            self.logger.error(f"Failed to convert batch results: {e}")

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
            self.logger.error(f"Failed to calculate statistics: {e}")
            return CrawlStatistics()

    def crawl_stats_to_statistics(self, stats: CrawlStats) -> CrawlStatistics:
        """
        Convert parallel engine CrawlStats to our CrawlStatistics model.

        Args:
            stats: CrawlStats from parallel engine

        Returns:
            CrawlStatistics compatible with our system
        """
        return CrawlStatistics(
            total_pages_requested=stats.urls_requested,
            total_pages_crawled=stats.urls_successful,
            total_pages_failed=stats.urls_failed,
            unique_domains=1,  # Would need additional info to calculate
            total_links_discovered=0,  # Would need additional info to calculate
            total_bytes_downloaded=stats.total_content_length,
            crawl_duration_seconds=stats.total_duration,
            pages_per_second=stats.pages_per_second,
            average_page_size=stats.average_content_length,
        )

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
            timestamp=datetime.now(),
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
