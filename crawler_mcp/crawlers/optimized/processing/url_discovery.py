"""
URL discovery system for optimized high-performance web crawler.

This module discovers URLs from sitemaps and robots.txt without initial crawling,
enabling intelligent URL prioritization and efficient crawl planning.
"""

import asyncio
import gzip
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp

from ..config import OptimizedConfig


class URLDiscovery:
    """Discovers URLs from sitemaps and robots.txt without crawling"""

    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize URL discovery system.

        Args:
            config: Optional optimized crawler configuration
        """
        self.config = config or OptimizedConfig()
        self.logger = logging.getLogger(__name__)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()

    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=getattr(
                    self.config,
                    "sitemap_timeout_seconds",
                    self.config.discovery_timeout,
                )
            )
            connector = aiohttp.TCPConnector(
                limit=20, ttl_dns_cache=300, enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": getattr(
                        self.config,
                        "sitemap_user_agent",
                        "Mozilla/5.0 (compatible; OptimizedCrawler/1.0)",
                    ),
                    "Accept": "application/xml, text/xml, application/rss+xml, application/atom+xml, text/plain, */*",
                    "Accept-Encoding": "gzip, deflate"
                    if getattr(self.config, "sitemap_accept_compressed", True)
                    else "identity",
                },
                connector_owner=True,
                auto_decompress=getattr(self.config, "sitemap_accept_compressed", True),
            )

    async def _close_session(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def discover_all(
        self, start_url: str, max_urls: int | None = None
    ) -> list[str]:
        """
        Main discovery method that finds URLs without crawling.

        This method combines sitemap discovery, robots.txt parsing, and URL scoring
        to return a prioritized list of URLs for crawling.

        Args:
            start_url: Starting URL to discover from
            max_urls: Maximum URLs to return (defaults to config value)

        Returns:
            List of discovered URLs sorted by relevance score
        """
        max_urls = max_urls or self.config.max_urls_to_discover

        # Ensure session is available
        await self._ensure_session()

        discovery_stats = {
            "start_url": start_url,
            "max_urls_requested": max_urls,
            "sources_enabled": [],
            "sources_results": {},
            "total_discovered": 0,
            "filtered_count": 0,
            "final_count": 0,
            "errors": [],
        }

        try:
            # Parse the starting URL
            parsed_url = urlparse(start_url)
            domain = parsed_url.netloc
            base_url = f"{parsed_url.scheme}://{domain}"

            self.logger.info(
                f"🔍 Starting URL discovery for domain: {domain} (max: {max_urls})"
            )

            # Track enabled sources
            if self.config.discover_from_sitemap:
                discovery_stats["sources_enabled"].append("direct_sitemap")
            if self.config.discover_from_robots:
                discovery_stats["sources_enabled"].append("robots_sitemaps")
            discovery_stats["sources_enabled"].append("common_locations")

            # Discover URLs from multiple sources concurrently
            all_urls: set[str] = set()

            tasks = []
            source_labels = []

            if self.config.discover_from_sitemap:
                tasks.append(self._discover_from_sitemap(base_url))
                source_labels.append("direct_sitemap")
            if self.config.discover_from_robots:
                tasks.append(self._discover_from_robots(base_url))
                source_labels.append("robots_sitemaps")
            tasks.append(self._discover_from_common_locations(base_url))
            source_labels.append("common_locations")

            self.logger.info(
                f"📡 Querying {len(tasks)} discovery sources: {', '.join(source_labels)}"
            )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for _i, (result, label) in enumerate(
                zip(results, source_labels, strict=False)
            ):
                if isinstance(result, list):
                    source_count = len(result)
                    all_urls.update(result)
                    discovery_stats["sources_results"][label] = {
                        "count": source_count,
                        "error": None,
                    }
                    if source_count > 0:
                        self.logger.warning(
                            f"✅ Found {source_count} URLs from {label}"
                        )
                    else:
                        self.logger.warning(f"⚠️  No URLs found from {label}")
                elif isinstance(result, Exception):
                    error_msg = str(result)
                    discovery_stats["sources_results"][label] = {
                        "count": 0,
                        "error": error_msg,
                    }
                    discovery_stats["errors"].append(f"{label}: {error_msg}")
                    self.logger.error(f"❌ Error in {label}: {error_msg}")
                else:
                    discovery_stats["sources_results"][label] = {
                        "count": 0,
                        "error": f"Unexpected result type: {type(result)}",
                    }
                    self.logger.warning(
                        f"⚠️  Unexpected result from {label}: {type(result)}"
                    )

            # Convert to list and filter
            url_list = list(all_urls)
            discovery_stats["total_discovered"] = len(url_list)

            if not url_list:
                self.logger.warning(
                    f"🚨 No URLs discovered from any source for {domain}"
                )
                discovery_stats["final_count"] = 1
                return [start_url]  # Return original URL as fallback

            self.logger.info(f"📊 Total unique URLs discovered: {len(url_list)}")

            # Filter URLs by domain and basic validation
            filtered_urls = self._filter_urls(url_list, domain)
            discovery_stats["filtered_count"] = len(filtered_urls)

            filtered_out = len(url_list) - len(filtered_urls)
            if filtered_out > 0:
                self.logger.info(
                    f"🔽 Filtered out {filtered_out} invalid URLs, {len(filtered_urls)} remain"
                )

            # Score and rank URLs
            scored_urls = await self._score_urls(filtered_urls, start_url)

            # Apply score threshold and limit
            final_urls = [
                url
                for url, score in scored_urls
                if score >= self.config.url_score_threshold
            ][:max_urls]

            discovery_stats["final_count"] = len(final_urls)

            score_filtered_out = len(filtered_urls) - len(final_urls)
            if score_filtered_out > 0:
                self.logger.info(
                    f"🎯 Applied scoring filter: removed {score_filtered_out} low-scoring URLs"
                )

            self.logger.info(
                f"🎉 Discovery completed: {len(final_urls)} URLs selected from {len(url_list)} discovered"
            )

            # Log top URLs for debugging
            if final_urls and len(final_urls) > 1:
                sample_size = min(5, len(final_urls))
                self.logger.debug(
                    f"🔗 Top {sample_size} URLs: {final_urls[:sample_size]}"
                )

            return final_urls

        except Exception as e:
            discovery_stats["errors"].append(f"discovery_main: {e!s}")
            self.logger.error(
                f"💥 URL discovery failed for {start_url}: {e}", exc_info=True
            )
            self.logger.info(f"🔄 Falling back to original URL: {start_url}")
            discovery_stats["final_count"] = 1
            return [start_url]
        finally:
            # Log comprehensive discovery statistics
            self._log_discovery_summary(discovery_stats)

    async def _discover_from_sitemap(self, base_url: str) -> list[str]:
        """Discover URLs from standard sitemap.xml"""
        sitemap_url = urljoin(base_url, "/sitemap.xml")
        return await self._fetch_sitemap(sitemap_url)

    async def _discover_from_robots(self, base_url: str) -> list[str]:
        """Discover URLs from robots.txt sitemap declarations"""
        robots_url = urljoin(base_url, "/robots.txt")

        try:
            async with self._session.get(robots_url) as response:
                if response.status != 200:
                    return []

                robots_text = await response.text()
                sitemap_urls = self._parse_robots_sitemaps(robots_text)

                # Fetch sitemaps concurrently with a small cap
                sem = asyncio.Semaphore(10)

                async def _fetch_with_sem(url: str) -> list[str]:
                    async with sem:
                        return await self._fetch_sitemap(url)

                tasks = [_fetch_with_sem(u) for u in sitemap_urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                all_urls: list[str] = []
                for r in results:
                    if isinstance(r, list):
                        all_urls.extend(r)

                return all_urls

        except Exception as e:
            self.logger.debug(f"Failed to fetch robots.txt from {robots_url}: {e}")
            return []

    async def _discover_from_common_locations(self, base_url: str) -> list[str]:
        """Discover URLs from common sitemap locations"""
        common_paths = [
            "/sitemap_index.xml",
            "/sitemaps.xml",
            "/sitemap/sitemap.xml",
            "/sitemaps/sitemap.xml",
            "/wp-sitemap.xml",  # WordPress
            "/sitemap-index.xml",
            "/feeds/sitemap.xml",
        ]

        # Try each common location concurrently
        sem = asyncio.Semaphore(10)

        async def _fetch(path: str) -> list[str]:
            async with sem:
                sitemap_url = urljoin(base_url, path)
                urls = await self._fetch_sitemap(sitemap_url)
                if urls:
                    self.logger.debug(f"Found {len(urls)} URLs at {path}")
                return urls

        results = await asyncio.gather(
            *[_fetch(p) for p in common_paths], return_exceptions=True
        )
        ret: list[str] = []
        for r in results:
            if isinstance(r, list):
                ret.extend(r)
        return ret

    async def _fetch_sitemap(self, sitemap_url: str) -> list[str]:
        """
        Fetch and parse a sitemap XML file with enhanced error handling and format support.

        Args:
            sitemap_url: URL of the sitemap to fetch

        Returns:
            List of URLs found in the sitemap
        """
        try:
            follow_redirects = getattr(self.config, "sitemap_follow_redirects", True)
            max_redirects = getattr(self.config, "sitemap_max_redirects", 5)

            async with self._session.get(
                sitemap_url,
                allow_redirects=follow_redirects,
                max_redirects=max_redirects,
            ) as response:
                # Accept various success codes including redirects
                if response.status not in (200, 301, 302):
                    self.logger.warning(
                        f"Sitemap fetch failed: {sitemap_url} returned {response.status}"
                    )
                    return []

                # Check content type for basic validation
                content_type = response.headers.get("content-type", "").lower()
                self.logger.debug(
                    f"Sitemap content-type: {content_type} for {sitemap_url}"
                )

                # Handle compressed content - check if actually gzipped
                content_encoding = response.headers.get("content-encoding", "").lower()
                is_gzipped_file = sitemap_url.endswith(".gz")
                is_gzipped_encoding = "gzip" in content_encoding

                if is_gzipped_file or is_gzipped_encoding:
                    raw_content = await response.read()
                    try:
                        # Check if content is actually gzipped by looking at magic bytes
                        if len(raw_content) >= 2 and raw_content[:2] == b"\x1f\x8b":
                            xml_content = gzip.decompress(raw_content).decode("utf-8")
                            self.logger.debug(
                                f"Decompressed gzipped sitemap: {sitemap_url}"
                            )
                        else:
                            # Not actually gzipped, treat as regular text
                            xml_content = raw_content.decode("utf-8")
                            self.logger.debug(
                                f"File had .gz extension but wasn't gzipped: {sitemap_url}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to process sitemap content {sitemap_url}: {e}"
                        )
                        return []
                else:
                    xml_content = await response.text()

                return await self._parse_sitemap_xml(xml_content, sitemap_url)

        except TimeoutError:
            self.logger.warning(f"Timeout fetching sitemap {sitemap_url}")
            return []
        except Exception as e:
            self.logger.warning(f"Failed to fetch sitemap {sitemap_url}: {e}")
            return []

    async def _parse_sitemap_xml(self, xml_content: str, sitemap_url: str) -> list[str]:
        """
        Parse sitemap XML content and extract URLs with enhanced namespace handling.

        Args:
            xml_content: Raw XML content of the sitemap
            sitemap_url: URL of the sitemap (for relative URL resolution)

        Returns:
            List of URLs found in the sitemap
        """
        if not xml_content or not xml_content.strip():
            self.logger.warning(f"Empty sitemap content from {sitemap_url}")
            return []

        # Try different parsing approaches
        urls = await self._try_xml_parsing_strategies(xml_content, sitemap_url)

        if urls:
            self.logger.warning(
                f"🗺️ Successfully extracted {len(urls)} URLs from sitemap {sitemap_url}"
            )
        else:
            # Try alternate formats as fallback
            urls = await self._try_alternate_formats(xml_content, sitemap_url)

        return urls

    async def _try_xml_parsing_strategies(
        self, xml_content: str, sitemap_url: str
    ) -> list[str]:
        """Try multiple XML parsing strategies for maximum compatibility."""
        strategies = [
            self._parse_with_namespaces,
            self._parse_without_namespaces,
            self._parse_with_manual_extraction,
        ]

        for strategy in strategies:
            try:
                urls = await strategy(xml_content, sitemap_url)
                if urls:
                    self.logger.debug(
                        f"Successful parsing with {strategy.__name__} for {sitemap_url}"
                    )
                    return urls
            except Exception as e:
                self.logger.debug(
                    f"Strategy {strategy.__name__} failed for {sitemap_url}: {e}"
                )
                continue

        return []

    async def _parse_with_namespaces(
        self, xml_content: str, sitemap_url: str
    ) -> list[str]:
        """Parse XML with proper namespace handling."""
        try:
            root = ET.fromstring(xml_content)

            # Register common sitemap namespaces
            namespaces = {
                "sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9",
                "news": "http://www.google.com/schemas/sitemap-news/0.9",
                "image": "http://www.google.com/schemas/sitemap-image/1.1",
                "video": "http://www.google.com/schemas/sitemap-video/1.1",
            }

            # Auto-detect namespace from root element
            if root.tag.startswith("{"):
                namespace_uri = root.tag.split("}")[0][1:]
                namespaces["default"] = namespace_uri

            urls = []

            if self._is_sitemap_index_enhanced(root):
                # This is a sitemap index
                sitemap_urls = self._extract_sitemap_urls_enhanced(root, namespaces)

                # Fetch child sitemaps with conservative limits
                max_children = 200
                sem = asyncio.Semaphore(10)

                async def _fetch_child(u: str) -> list[str]:
                    async with sem:
                        return await self._fetch_sitemap(u)

                tasks = [_fetch_child(u) for u in sitemap_urls[:max_children]]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, list):
                        urls.extend(result)
            else:
                # Regular sitemap
                urls = self._extract_page_urls_enhanced(root, namespaces)

            return urls

        except ET.ParseError as e:
            self.logger.warning(f"XML parse error in sitemap {sitemap_url}: {e}")
            raise
        except Exception as e:
            self.logger.warning(f"Namespace parsing failed for {sitemap_url}: {e}")
            raise

    async def _parse_without_namespaces(
        self, xml_content: str, sitemap_url: str
    ) -> list[str]:
        """Parse XML ignoring namespaces (fallback method)."""
        try:
            # Remove namespace declarations for simpler parsing
            cleaned_content = xml_content
            import re

            # Remove xmlns declarations
            cleaned_content = re.sub(r'\s+xmlns[^=]*="[^"]*"', "", cleaned_content)
            # Remove namespace prefixes from tags
            cleaned_content = re.sub(r"</?\w+:", "<", cleaned_content)

            root = ET.fromstring(cleaned_content)
            urls = []

            # Check for sitemap index
            if root.tag.lower().endswith("sitemapindex"):
                # Extract sitemap URLs
                for sitemap_elem in root.findall(".//sitemap"):
                    loc_elem = sitemap_elem.find(".//loc")
                    if loc_elem is not None and loc_elem.text:
                        urls.append(loc_elem.text.strip())

                # Fetch child sitemaps
                sem = asyncio.Semaphore(10)

                async def _fetch_child(u: str) -> list[str]:
                    async with sem:
                        return await self._fetch_sitemap(u)

                tasks = [_fetch_child(u) for u in urls[:200]]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                child_urls = []
                for result in results:
                    if isinstance(result, list):
                        child_urls.extend(result)
                return child_urls
            else:
                # Extract page URLs
                for url_elem in root.findall(".//url"):
                    loc_elem = url_elem.find(".//loc")
                    if loc_elem is not None and loc_elem.text:
                        urls.append(loc_elem.text.strip())

            return urls

        except Exception as e:
            self.logger.debug(f"Namespace-free parsing failed for {sitemap_url}: {e}")
            raise

    async def _parse_with_manual_extraction(
        self, xml_content: str, sitemap_url: str
    ) -> list[str]:
        """Manual regex-based URL extraction as last resort."""
        try:
            import re

            urls = []

            # Extract URLs from <loc> tags using regex
            loc_pattern = r"<loc[^>]*>\s*([^<]+)\s*</loc>"
            matches = re.findall(loc_pattern, xml_content, re.IGNORECASE)

            for match in matches:
                url = match.strip()
                if url and url.startswith(("http://", "https://")):
                    urls.append(url)

            self.logger.debug(
                f"Manual extraction found {len(urls)} URLs in {sitemap_url}"
            )
            return urls

        except Exception as e:
            self.logger.warning(f"Manual extraction failed for {sitemap_url}: {e}")
            raise

    async def _try_alternate_formats(self, content: str, sitemap_url: str) -> list[str]:
        """Try parsing as RSS/Atom or text format."""
        try:
            # Try RSS/Atom format
            rss_urls = await self._parse_rss_atom(content, sitemap_url)
            if rss_urls:
                return rss_urls

            # Try plain text format
            text_urls = await self._parse_text_sitemap(content, sitemap_url)
            if text_urls:
                return text_urls

        except Exception as e:
            self.logger.debug(f"Alternate format parsing failed for {sitemap_url}: {e}")

        return []

    async def _parse_rss_atom(self, content: str, sitemap_url: str) -> list[str]:
        """Parse RSS or Atom feeds as sitemaps."""
        try:
            root = ET.fromstring(content)
            urls = []

            # RSS format
            if "rss" in root.tag.lower():
                for item in root.findall(".//item"):
                    link_elem = item.find(".//link")
                    if link_elem is not None and link_elem.text:
                        urls.append(link_elem.text.strip())

            # Atom format
            elif "feed" in root.tag.lower():
                for entry in root.findall(".//entry"):
                    for link_elem in entry.findall(".//link"):
                        href = link_elem.get("href")
                        if href:
                            urls.append(href.strip())

            if urls:
                self.logger.info(
                    f"Parsed {len(urls)} URLs from RSS/Atom format: {sitemap_url}"
                )
            return urls

        except Exception as e:
            self.logger.debug(f"RSS/Atom parsing failed for {sitemap_url}: {e}")
            return []

    async def _parse_text_sitemap(self, content: str, sitemap_url: str) -> list[str]:
        """Parse plain text sitemap (one URL per line)."""
        try:
            urls = []
            for line in content.splitlines():
                line = line.strip()
                if line and line.startswith(("http://", "https://")):
                    urls.append(line)

            if urls:
                self.logger.info(
                    f"Parsed {len(urls)} URLs from text format: {sitemap_url}"
                )
            return urls

        except Exception as e:
            self.logger.debug(f"Text parsing failed for {sitemap_url}: {e}")
            return []

    def _is_sitemap_index(self, root: ET.Element) -> bool:
        """Check if XML root is a sitemap index (legacy method for compatibility)"""
        return self._is_sitemap_index_enhanced(root)

    def _is_sitemap_index_enhanced(self, root: ET.Element) -> bool:
        """Enhanced check if XML root is a sitemap index with better detection"""
        tag_name = root.tag.lower()
        # Remove namespace prefix if present
        if "}" in tag_name:
            tag_name = tag_name.split("}", 1)[1]
        return tag_name in ("sitemapindex", "sitemap_index")

    def _extract_sitemap_urls(self, root: ET.Element) -> list[str]:
        """Extract sitemap URLs from sitemap index (legacy method)"""
        return self._extract_sitemap_urls_enhanced(root, {})

    def _extract_sitemap_urls_enhanced(
        self, root: ET.Element, namespaces: dict
    ) -> list[str]:
        """Extract sitemap URLs from sitemap index with namespace support"""
        urls = []

        # Try with namespaces first
        for ns_prefix, ns_uri in namespaces.items():
            try:
                if ns_prefix == "default":
                    # Use default namespace
                    xpath = f".//{{{ns_uri}}}sitemap/{{{ns_uri}}}loc"
                else:
                    # Use named namespace
                    xpath = f".//{ns_prefix}:sitemap/{ns_prefix}:loc"

                for loc_elem in root.findall(xpath, namespaces):
                    if loc_elem is not None and loc_elem.text:
                        urls.append(loc_elem.text.strip())

                if urls:
                    return urls
            except Exception:
                continue

        # Fallback to namespace-agnostic search
        for sitemap in root.findall(".//{*}sitemap"):
            loc_elem = sitemap.find(".//{*}loc")
            if loc_elem is not None and loc_elem.text:
                urls.append(loc_elem.text.strip())

        # Final fallback - search without namespace wildcards
        if not urls:
            for sitemap in root.iter():
                if sitemap.tag.endswith("sitemap") or "sitemap" in sitemap.tag.lower():
                    for child in sitemap:
                        if child.tag.endswith("loc") and child.text:
                            urls.append(child.text.strip())

        return urls

    def _extract_page_urls(self, root: ET.Element) -> list[str]:
        """Extract page URLs from regular sitemap (legacy method)"""
        return self._extract_page_urls_enhanced(root, {})

    def _extract_page_urls_enhanced(
        self, root: ET.Element, namespaces: dict
    ) -> list[str]:
        """Extract page URLs from regular sitemap with namespace support"""
        urls = []

        # Try with namespaces first
        for ns_prefix, ns_uri in namespaces.items():
            try:
                if ns_prefix == "default":
                    xpath = f".//{{{ns_uri}}}url/{{{ns_uri}}}loc"
                else:
                    xpath = f".//{ns_prefix}:url/{ns_prefix}:loc"

                for loc_elem in root.findall(xpath, namespaces):
                    if loc_elem is not None and loc_elem.text:
                        urls.append(loc_elem.text.strip())

                if urls:
                    return urls
            except Exception:
                continue

        # Fallback to namespace-agnostic search
        for url_elem in root.findall(".//{*}url"):
            loc_elem = url_elem.find(".//{*}loc")
            if loc_elem is not None and loc_elem.text:
                urls.append(loc_elem.text.strip())

        # Final fallback - search without namespace wildcards
        if not urls:
            for url_elem in root.iter():
                if url_elem.tag.endswith("url") or "url" in url_elem.tag.lower():
                    for child in url_elem:
                        if child.tag.endswith("loc") and child.text:
                            urls.append(child.text.strip())

        return urls

    def _parse_robots_sitemaps(self, robots_text: str) -> list[str]:
        """
        Parse robots.txt content and extract sitemap URLs.

        Note: We extract sitemap URLs but ignore all other robots.txt rules
        as requested by the user.

        Args:
            robots_text: Content of robots.txt file

        Returns:
            List of sitemap URLs found in robots.txt
        """
        sitemaps = []

        for line in robots_text.splitlines():
            line = line.strip()

            # Look for sitemap declarations (case insensitive)
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                if sitemap_url:
                    sitemaps.append(sitemap_url)

        return sitemaps

    def _filter_urls(self, urls: list[str], domain: str) -> list[str]:
        """
        Filter URLs to remove invalid or unwanted URLs.

        Args:
            urls: List of URLs to filter
            domain: Target domain to filter for

        Returns:
            List of filtered, valid URLs
        """
        filtered = []

        self.logger.debug(f"🔍 Filtering {len(urls)} URLs for domain: {domain}")
        for i, url in enumerate(urls):
            is_valid = self._is_valid_url(url, domain)
            is_allowed = self._is_allowed_locale(url)

            if is_valid and is_allowed:
                filtered.append(url)
                self.logger.debug(f"  ✅ {i + 1}. {url}")
            else:
                reason = []
                if not is_valid:
                    reason.append("invalid")
                if not is_allowed:
                    reason.append("locale")
                self.logger.debug(
                    f"  ❌ {i + 1}. {url} - filtered ({', '.join(reason)})"
                )

        # Normalize and remove duplicates while preserving order
        seen = set()
        deduped: list[str] = []
        for u in filtered:
            nu = self._normalize_url(u)
            if nu not in seen:
                seen.add(nu)
                deduped.append(nu)

        return deduped

    def _is_valid_url(self, url: str, domain: str) -> bool:
        """
        Check if URL is valid for crawling.

        Args:
            url: URL to validate
            domain: Target domain

        Returns:
            True if URL is valid for crawling
        """
        try:
            parsed = urlparse(url)

            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False

            # Must match target domain
            if parsed.netloc != domain:
                return False

            # Skip common non-content file types
            excluded_extensions = {
                ".pdf",
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".svg",
                ".mp4",
                ".mp3",
                ".avi",
                ".mov",
                ".zip",
                ".tar",
                ".gz",
                ".css",
                ".js",
                ".ico",
                ".xml",
                ".json",
                ".txt",
            }

            path = parsed.path.lower()
            if any(path.endswith(ext) for ext in excluded_extensions):
                return False

            # Skip URLs with too many path segments (likely dynamic)
            if len(parsed.path.split("/")) > 10:
                return False

            # Skip URLs with query parameters that look like session IDs
            if parsed.query:
                query_lower = parsed.query.lower()
                session_indicators = ["sessionid", "sid", "jsessionid", "phpsessid"]
                if any(indicator in query_lower for indicator in session_indicators):
                    return False

            return True

        except Exception:
            return False

    def _normalize_url(self, url: str) -> str:
        """Canonicalize URL for deduplication: remove fragments and normalize trailing slash."""
        try:
            parsed = urlparse(url)
            # Remove fragment and normalize path trailing slash for directory-like paths
            path = parsed.path or "/"
            if not path.endswith("/") and "." not in (path.split("/")[-1] or ""):
                path = path + "/"
            normalized = parsed._replace(fragment="", path=path)
            return urlunparse(normalized)
        except Exception:
            return url

    async def _score_urls(
        self, urls: list[str], base_url: str
    ) -> list[tuple[str, float]]:
        """
        Score and rank URLs by relevance.

        Args:
            urls: List of URLs to score
            base_url: Base URL for relevance comparison

        Returns:
            List of (url, score) tuples sorted by score (highest first)
        """
        scored_urls = []
        base_parsed = urlparse(base_url)
        base_path = base_parsed.path

        for url in urls:
            score = self._calculate_url_score(url, base_url, base_path)
            scored_urls.append((url, score))

        # Sort by score (highest first)
        scored_urls.sort(key=lambda x: x[1], reverse=True)

        return scored_urls

    def _calculate_url_score(self, url: str, base_url: str, base_path: str) -> float:
        """
        Calculate relevance score for a URL.

        Args:
            url: URL to score
            base_url: Base URL for comparison
            base_path: Base path for comparison

        Returns:
            Relevance score (higher = more relevant)
        """
        try:
            parsed = urlparse(url)
            score = 1.0  # Base score

            # Prefer shorter paths (closer to root)
            path_depth = len([p for p in parsed.path.split("/") if p])
            score += max(0, 5 - path_depth) * 0.5

            # Prefer paths similar to base path
            if base_path and base_path in parsed.path:
                score += 2.0

            # Boost common content indicators
            content_indicators = [
                "article",
                "post",
                "blog",
                "news",
                "page",
                "content",
                "doc",
                "guide",
                "tutorial",
                "help",
                "about",
                "service",
            ]

            path_lower = parsed.path.lower()
            for indicator in content_indicators:
                if indicator in path_lower:
                    score += 1.0
                    break

            # Penalize dynamic-looking URLs
            if parsed.query:
                score -= 0.5

            # Penalize very long URLs
            if len(url) > 100:
                score -= 1.0

            # Penalize URLs with many numeric segments (often IDs)
            numeric_segments = sum(
                1 for part in parsed.path.split("/") if part.isdigit()
            )
            score -= numeric_segments * 0.3

            # Boost homepage and main sections
            if parsed.path in ["/", "/index.html", "/home"]:
                score += 3.0

            return max(0.0, score)  # Ensure non-negative score

        except Exception:
            return 0.1  # Very low score for problematic URLs

    def _is_allowed_locale(self, url: str) -> bool:
        """Check URL path locale against allowed locales if configured."""
        try:
            allowed = [s.lower() for s in (self.config.allowed_locales or [])]
            if not allowed:
                return True
            parsed = urlparse(url)
            segs = [s for s in parsed.path.split("/") if s]

            if not segs:
                # No path segments; allow if '' or 'en' included
                return ("" in allowed) or ("en" in allowed)

            first_seg = segs[0].lower()

            # Skip version numbers and other non-locale patterns
            if self._is_version_number(first_seg) or self._is_non_locale_segment(
                first_seg
            ):
                # No locale prefix, treat as default
                return ("" in allowed) or ("en" in allowed)

            # Check if first segment is a locale
            if first_seg in allowed:
                return True
            for a in allowed:
                if a and (first_seg == a or first_seg.startswith(a + "-")):
                    return True

            # If first segment doesn't look like a locale, treat as no-locale
            if not self._looks_like_locale(first_seg):
                return ("" in allowed) or ("en" in allowed)

            return False
        except Exception:
            return True

    def _is_version_number(self, segment: str) -> bool:
        """Check if segment looks like a version number."""
        # Match patterns like: 3, 3.10, 3.11, v1.0, 2.7, etc.
        import re

        version_patterns = [
            r"^\d+$",  # Just a number: 3
            r"^\d+\.\d+$",  # Major.minor: 3.10
            r"^\d+\.\d+\.\d+$",  # Major.minor.patch: 3.10.1
            r"^v?\d+(\.\d+)*$",  # With optional v prefix: v3.10
        ]
        return any(re.match(pattern, segment) for pattern in version_patterns)

    def _is_non_locale_segment(self, segment: str) -> bool:
        """Check if segment is clearly not a locale code."""
        non_locale_segments = {
            "docs",
            "api",
            "documentation",
            "guide",
            "tutorial",
            "reference",
            "manual",
            "help",
            "faq",
            "blog",
            "news",
            "download",
            "install",
            "setup",
            "config",
            "admin",
            "user",
            "dev",
            "developer",
        }
        return segment in non_locale_segments

    def _looks_like_locale(self, segment: str) -> bool:
        """Check if segment looks like a locale code."""
        # Locale codes are typically 2-5 chars, contain letters, possibly with dashes
        import re

        # Match patterns like: en, fr, en-us, pt-br, zh-cn, etc.
        return (
            bool(re.match(r"^[a-z]{2}(-[a-z]{2,3})?$", segment)) and len(segment) <= 5
        )

    def _log_discovery_summary(self, stats: dict) -> None:
        """Log comprehensive discovery statistics for debugging."""
        try:
            self.logger.warning("📈 URL Discovery Summary:")
            self.logger.info(f"   🎯 Target: {stats['start_url']}")
            self.logger.info(f"   🔢 Requested: {stats['max_urls_requested']} URLs")
            self.logger.info(f"   📡 Sources: {', '.join(stats['sources_enabled'])}")

            # Source breakdown
            for source, result in stats["sources_results"].items():
                if result["error"]:
                    self.logger.warning(f"   ❌ {source}: ERROR - {result['error']}")
                else:
                    self.logger.warning(f"   ✅ {source}: {result['count']} URLs")

            self.logger.info(
                f"   📊 Pipeline: {stats['total_discovered']} → {stats['filtered_count']} → {stats['final_count']}"
            )

            if stats["errors"]:
                self.logger.warning(f"   ⚠️  Errors encountered: {len(stats['errors'])}")
                for error in stats["errors"][:3]:  # Show first 3 errors
                    self.logger.warning(f"      • {error}")

            # Success rate
            if stats["total_discovered"] > 0:
                success_rate = (stats["final_count"] / stats["total_discovered"]) * 100
                self.logger.info(
                    f"   📈 Success rate: {success_rate:.1f}% (final/discovered)"
                )

        except Exception as e:
            self.logger.debug(f"Failed to log discovery summary: {e}")

    async def get_domain_info(self, domain: str) -> dict[str, Any]:
        """
        Get information about a domain's URL structure.

        Args:
            domain: Domain to analyze

        Returns:
            Dictionary with domain information
        """
        base_url = f"https://{domain}"

        await self._ensure_session()

        # Discover URLs
        urls = await self.discover_all(base_url, max_urls=100)

        # Analyze URL patterns
        path_segments = defaultdict(int)
        file_types = defaultdict(int)

        for url in urls:
            parsed = urlparse(url)

            # Count path segments
            segments = [s for s in parsed.path.split("/") if s]
            for segment in segments:
                path_segments[segment] += 1

            # Count file extensions
            path = parsed.path.lower()
            if "." in path:
                ext = path.split(".")[-1]
                if len(ext) <= 5:  # Reasonable extension length
                    file_types[ext] += 1

        return {
            "domain": domain,
            "total_urls_discovered": len(urls),
            "common_path_segments": dict(
                sorted(path_segments.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "file_types": dict(
                sorted(file_types.items(), key=lambda x: x[1], reverse=True)
            ),
            "sample_urls": urls[:10],
        }
