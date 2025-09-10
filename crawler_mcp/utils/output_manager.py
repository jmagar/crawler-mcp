"""Output management for the optimized crawler.

Provides standardized output organization, rotation, and cleanup.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from crawler_mcp.optimized_config import OptimizedConfig


class OutputManager:
    """Manages all output files with automatic rotation and cleanup."""

    def __init__(
        self, base_dir: str = "./.crawl4ai", config: OptimizedConfig | None = None
    ):
        """Initialize with base directory and configuration.

        Args:
            base_dir: Base output directory path
            config: Crawler configuration for limits and settings
        """
        self.base_dir = Path(base_dir)
        self.config = config or OptimizedConfig()

        # Core directories
        self.crawls_dir = self.base_dir / "crawls"
        self.pr_dir = self.base_dir / "pr"
        self.cache_dir = self.base_dir / "cache"
        self.logs_dir = self.base_dir / "logs"

        # Index files
        self.crawls_index = self.crawls_dir / "_index.json"
        self.pr_registry = self.pr_dir / "_registry.json"

        # Ensure directories exist
        self._init_directories()

    def _init_directories(self) -> None:
        """Create all required directories."""
        for dir_path in [self.crawls_dir, self.pr_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create cache subdirectories
        (self.cache_dir / "search").mkdir(exist_ok=True)

    def sanitize_domain(self, url: str) -> str:
        """Convert URL to safe directory name.

        Args:
            url: URL to sanitize

        Returns:
            Sanitized domain name (example.com → example_com)
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split("/")[0]

            # Remove port if present
            if ":" in domain:
                domain = domain.split(":")[0]

            # Replace dots with underscores, convert to lowercase
            sanitized = domain.lower().replace(".", "_")

            # Remove any remaining special characters
            allowed_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
            sanitized = "".join(c for c in sanitized if c in allowed_chars)

            return sanitized or "unknown"
        except Exception:
            return "unknown"

    def get_crawl_output_paths(
        self, url: str, session_id: str | None = None
    ) -> dict[str, Path]:
        """Get all output paths for a crawl.

        Args:
            url: URL being crawled
            session_id: Unique session identifier. If None, uses "latest" for backward compatibility.

        Returns:
            Dictionary with output paths: html, ndjson, report
        """
        domain = self.sanitize_domain(url)

        # Use session_id if provided, otherwise fall back to "latest" for backward compatibility
        dir_name = session_id if session_id else "latest"
        session_dir = self.crawls_dir / domain / dir_name
        session_dir.mkdir(parents=True, exist_ok=True)

        return {
            "html": session_dir / "combined.html",
            "ndjson": session_dir / "pages.ndjson",
            "report": session_dir / "report.json",
        }

    def get_pr_output_paths(
        self, owner: str, repo: str, pr_number: int
    ) -> dict[str, Path]:
        """Get all output paths for PR operations.

        Args:
            owner: GitHub repository owner
            repo: GitHub repository name
            pr_number: Pull request number

        Returns:
            Dictionary with output paths: items, suggestions, resolved
        """
        pr_name = f"{repo}-{pr_number}"
        pr_dir = self.pr_dir / pr_name
        pr_dir.mkdir(parents=True, exist_ok=True)

        return {
            "items": pr_dir / "items.ndjson",
            "suggestions": pr_dir / "suggestions.json",
            "resolved": pr_dir / "resolved.json",
        }

    def rotate_crawl_backup(self, domain: str) -> None:
        """Prepare domain directory for new crawl session (no longer needed with session-based dirs).

        This method is kept for backward compatibility but no longer performs rotation
        since each crawl now gets a unique session directory. The 'latest' symlink
        will be updated to point to the new session.

        Args:
            domain: Sanitized domain name
        """
        # With session-based directories, we don't need to rotate anything
        # Each crawl gets its own unique directory and the 'latest' symlink
        # will be updated to point to the newest session
        pass

    def save_crawl_outputs(
        self,
        domain: str,
        html: str | None,
        pages: list[Any],
        report: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        """Save all crawl outputs with automatic rotation.

        Args:
            domain: Sanitized domain name
            html: Combined HTML content
            pages: List of page objects
            report: Performance report data
            session_id: Unique session identifier. If None, uses "latest" for backward compatibility.
        """
        paths = self.get_crawl_output_paths(f"https://{domain}", session_id)

        try:
            # Write HTML output
            if html:
                paths["html"].write_text(html, encoding="utf-8")

            # Write NDJSON pages
            with open(paths["ndjson"], "w", encoding="utf-8") as f:
                for page in pages:
                    # Convert page object to dict
                    if hasattr(page, "__dict__"):
                        page_dict = {
                            "url": getattr(page, "url", ""),
                            "title": getattr(page, "title", ""),
                            "word_count": getattr(page, "word_count", 0),
                            "links": getattr(page, "links", []),
                            "images": getattr(page, "images", []),
                            "metadata": getattr(page, "metadata", {}),
                            "content": getattr(page, "content", ""),
                        }
                    else:
                        page_dict = page

                    f.write(json.dumps(page_dict, ensure_ascii=False) + "\n")

            # Write report
            with open(paths["report"], "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        except Exception as e:
            # Log error but don't fail the crawl
            print(f"Warning: Failed to save outputs for {domain}: {e}")

    def create_latest_symlink(self, domain: str, session_id: str) -> None:
        """Create a 'latest' symlink pointing to the most recent session directory.

        Args:
            domain: Sanitized domain name
            session_id: Session identifier to link to
        """
        try:
            domain_dir = self.crawls_dir / domain
            latest_link = domain_dir / "latest"
            session_target = session_id

            # Remove existing symlink or directory
            if latest_link.exists() or latest_link.is_symlink():
                if latest_link.is_symlink():
                    latest_link.unlink()
                elif latest_link.is_dir():
                    shutil.rmtree(latest_link)

            # Create new symlink pointing to session directory
            latest_link.symlink_to(session_target, target_is_directory=True)

        except Exception as e:
            print(f"Warning: Failed to create latest symlink for {domain}: {e}")

    def update_index(
        self, domain: str, metadata: dict[str, Any], session_id: str | None = None
    ) -> None:
        """Update _index.json with crawl metadata.

        Args:
            domain: Sanitized domain name
            metadata: Crawl metadata to store
            session_id: Unique session identifier. If provided, tracks individual sessions.
        """
        try:
            # Load existing index
            if self.crawls_index.exists():
                with open(self.crawls_index, encoding="utf-8") as f:
                    index = json.load(f)
            else:
                index = {"domains": {}, "total_size_bytes": 0, "last_cleanup": 0}

            # Update domain entry
            if domain not in index["domains"]:
                index["domains"][domain] = {"sessions": {}}

            # If session_id is provided, track individual sessions
            if session_id:
                # Initialize sessions dict if it doesn't exist (backward compatibility)
                if "sessions" not in index["domains"][domain]:
                    index["domains"][domain]["sessions"] = {}

                # Add session-specific metadata
                index["domains"][domain]["sessions"][session_id] = {
                    **metadata,
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "size_bytes": self._get_domain_size(domain),
                }

                # Update latest pointer to most recent session
                index["domains"][domain]["latest_session"] = session_id

            # For backward compatibility, also update "latest" entry
            index["domains"][domain]["latest"] = {
                **metadata,
                "session_id": session_id,
                "timestamp": time.time(),
                "size_bytes": self._get_domain_size(domain),
            }

            # Update total size
            index["total_size_bytes"] = self.get_total_size()

            # Save index
            with open(self.crawls_index, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Warning: Failed to update index for {domain}: {e}")

    def _get_domain_size(self, domain: str) -> int:
        """Get total size of domain directory in bytes."""
        domain_dir = self.crawls_dir / domain
        if not domain_dir.exists():
            return 0

        total_size = 0
        for file_path in domain_dir.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except OSError:
                    continue
        return total_size

    def get_total_size(self) -> int:
        """Get total size of output directory in bytes."""
        if not self.base_dir.exists():
            return 0

        total_size = 0
        for file_path in self.base_dir.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except OSError:
                    continue
        return total_size

    def cleanup_old_outputs(self) -> None:
        """Remove old outputs if size limit exceeded."""
        total_size = self.get_total_size()
        max_size = getattr(self.config, "max_output_size_gb", 1.0) * 1024 * 1024 * 1024

        if total_size <= max_size:
            return

        print(
            f"Output size {total_size / (1024 * 1024):.1f}MB exceeds limit, cleaning up..."
        )

        try:
            # Load index to find oldest entries
            if not self.crawls_index.exists():
                return

            with open(self.crawls_index, encoding="utf-8") as f:
                index = json.load(f)

            # Remove backup directories first (oldest first)
            domains_by_backup_age = []
            for domain, data in index.get("domains", {}).items():
                if "backup" in data:
                    backup_time = data["backup"].get("timestamp", 0)
                    domains_by_backup_age.append((backup_time, domain))

            domains_by_backup_age.sort()  # Oldest first

            for _, domain in domains_by_backup_age:
                backup_dir = self.crawls_dir / domain / "backup"
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                    if domain in index["domains"]:
                        index["domains"][domain].pop("backup", None)

                    # Check if we're under the limit now
                    if self.get_total_size() <= max_size:
                        break

            # If still over limit, remove entire domain directories (oldest first)
            if self.get_total_size() > max_size:
                domains_by_latest_age = []
                for domain, data in index.get("domains", {}).items():
                    if "latest" in data:
                        latest_time = data["latest"].get("timestamp", 0)
                        domains_by_latest_age.append((latest_time, domain))

                domains_by_latest_age.sort()  # Oldest first

                for _, domain in domains_by_latest_age:
                    domain_dir = self.crawls_dir / domain
                    if domain_dir.exists():
                        shutil.rmtree(domain_dir)
                        index["domains"].pop(domain, None)

                    # Check if we're under the limit now
                    if self.get_total_size() <= max_size:
                        break

            # Update index
            index["total_size_bytes"] = self.get_total_size()
            index["last_cleanup"] = time.time()

            with open(self.crawls_index, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")

    def clean_cache(self, max_age_hours: int = 24) -> None:
        """Remove cache files older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours before deletion
        """
        cutoff_time = time.time() - (max_age_hours * 3600)

        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                except OSError:
                    continue

        # Remove empty directories
        for dir_path in self.cache_dir.rglob("*"):
            if dir_path.is_dir():
                try:
                    dir_path.rmdir()  # Only removes if empty
                except OSError:
                    continue

    def get_pr_registry_path(self) -> Path:
        """Get path to PR registry file."""
        return self.pr_registry

    def get_search_cache_path(self, query_hash: str) -> Path:
        """Get path for search result caching.

        Args:
            query_hash: Hash of search query

        Returns:
            Path to cache file
        """
        return self.cache_dir / "search" / f"{query_hash}.json"
