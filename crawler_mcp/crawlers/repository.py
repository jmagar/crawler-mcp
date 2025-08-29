"""
Repository crawling strategy with git cloning and adaptive batch processing.
"""

import asyncio
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..core.logging import get_logger
from ..models.crawl import (
    CrawlResult,
    CrawlStatistics,
    CrawlStatus,
)
from .base import BaseCrawlStrategy
from .directory import DirectoryCrawlStrategy, DirectoryRequest

logger = get_logger(__name__)


class RepositoryRequest:
    """Request object for repository crawling."""

    def __init__(
        self,
        repo_url: str,
        clone_path: str | None = None,
        file_patterns: list[str] | None = None,
        max_files: int = 1000,
        cleanup_after: bool = True,
    ) -> None:
        self.repo_url = repo_url
        self.clone_path = clone_path
        self.file_patterns = file_patterns or [
            "*.py",
            "*.js",
            "*.ts",
            "*.md",
            "*.txt",
            "*.json",
            "*.yaml",
            "*.yml",
        ]
        self.max_files = max_files
        self.cleanup_after = cleanup_after


class RepositoryCrawlStrategy(BaseCrawlStrategy):
    """
    Repository analysis with git cloning and adaptive batch processing.
    Optimized for code repositories with intelligent file prioritization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.directory_strategy = DirectoryCrawlStrategy()

    async def validate_request(self, request: RepositoryRequest) -> bool:
        """Validate repository crawl request."""
        if not request.repo_url:
            self.logger.error("Repository URL is required")
            return False

        # Basic URL validation
        parsed = urlparse(request.repo_url)
        if not parsed.scheme or not parsed.netloc:
            self.logger.error(f"Invalid repository URL format: {request.repo_url}")
            return False

        # Check git availability
        if not await self._check_git_available():
            self.logger.error("Git is not available on the system")
            return False

        return True

    async def _check_git_available(self) -> bool:
        """Check if git command is available."""
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            return process.returncode == 0
        except FileNotFoundError:
            return False

    async def execute(
        self,
        request: RepositoryRequest,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """Execute repository crawling with cloning and processing."""
        await self._initialize_managers()

        self.logger.info(f"Starting repository crawl: {request.repo_url}")

        await self.pre_execute_setup()

        clone_dir = None
        try:
            # Clone repository
            if progress_callback:
                progress_callback(0, 100, "Cloning repository...")

            clone_dir = await self._clone_repository(request)

            if not clone_dir or not clone_dir.exists():
                raise Exception(f"Failed to clone repository: {request.repo_url}")

            if progress_callback:
                progress_callback(20, 100, "Repository cloned, analyzing files...")

            # Create directory request for processing
            dir_request = DirectoryRequest(
                directory_path=str(clone_dir),
                file_patterns=request.file_patterns,
                recursive=True,
                max_files=request.max_files,
            )

            # Process repository as directory with progress callback wrapper
            def wrapped_progress_callback(
                current: int, total: int, message: str | None = None
            ) -> None:
                if progress_callback:
                    # Map directory progress to 20-100 range
                    adjusted_progress = 20 + int((current / total) * 80)
                    progress_callback(adjusted_progress, 100, message)

            result = await self.directory_strategy.execute(
                dir_request, wrapped_progress_callback
            )

            # Enhance result with repository-specific information
            result = await self._enhance_result_with_repo_info(
                result, request, clone_dir
            )

            # Update request ID to indicate repository crawl
            result.request_id = f"repo_crawl_{int(time.time())}"
            result.urls = [request.repo_url]

            self.logger.info(
                f"Repository crawl completed: {len(result.pages)} files processed, "
                f"{result.statistics.crawl_duration_seconds:.1f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Repository crawl failed: {e}")

            return CrawlResult(
                request_id=f"repo_crawl_failed_{int(time.time())}",
                status=CrawlStatus.FAILED,
                urls=[request.repo_url],
                pages=[],
                errors=[str(e)],
                statistics=CrawlStatistics(),
            )

        finally:
            # Cleanup cloned repository if requested
            if clone_dir and request.cleanup_after:
                try:
                    if clone_dir.exists():
                        shutil.rmtree(clone_dir)
                        self.logger.debug(f"Cleaned up cloned repository: {clone_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup clone directory: {e}")

            await self.post_execute_cleanup()

    async def _clone_repository(self, request: RepositoryRequest) -> Path | None:
        """Clone repository to local directory."""
        try:
            # Determine clone directory
            if request.clone_path:
                clone_dir = Path(request.clone_path)
                clone_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Create temporary directory
                temp_dir = tempfile.mkdtemp(prefix="crawler_mcp_repo_")
                clone_dir = Path(temp_dir)

            repo_name = self._extract_repo_name(request.repo_url)
            target_dir = clone_dir / repo_name

            # Remove existing directory if it exists
            if target_dir.exists():
                shutil.rmtree(target_dir)

            self.logger.info(f"Cloning {request.repo_url} to {target_dir}")

            # Git clone command with optimizations
            cmd = [
                "git",
                "clone",
                "--depth",
                "1",  # Shallow clone for speed
                "--single-branch",  # Only clone default branch
                "--no-tags",  # Skip tags for speed
                request.repo_url,
                str(target_dir),
            ]

            # Execute git clone with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300,  # 5 minute timeout
            )

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown git error"
                raise Exception(f"Git clone failed: {error_msg}")

            if not target_dir.exists():
                raise Exception("Clone directory was not created")

            # Remove .git directory to save space
            git_dir = target_dir / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)

            self.logger.info(f"Successfully cloned repository to {target_dir}")
            return target_dir

        except TimeoutError as e:
            raise Exception("Repository cloning timed out (5 minutes)") from e
        except Exception as e:
            self.logger.error(f"Failed to clone repository {request.repo_url}: {e}")
            return None

    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL."""
        # Handle different URL formats
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]

        # Extract name from various URL formats
        name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

        # Clean up the name
        name = name.replace(":", "_").replace("@", "_")

        return name or "repository"

    async def _enhance_result_with_repo_info(
        self, result: CrawlResult, request: RepositoryRequest, clone_dir: Path
    ) -> CrawlResult:
        """Enhance crawl result with repository-specific metadata."""
        try:
            repo_info = await self._gather_repo_info(clone_dir)

            # Add repository metadata to each page
            for page in result.pages:
                if "repository_info" not in page.metadata:
                    page.metadata["repository_info"] = repo_info
                    page.metadata["repository_url"] = request.repo_url

            # Add repository info to result warnings for now
            # (CrawlResult doesn't have metadata field)
            result.warnings.append(
                f"Repository info: {repo_info['name']} - {repo_info['file_count']} files"
            )

            return result

        except Exception as e:
            self.logger.warning(f"Failed to gather repository info: {e}")
            return result

    async def _gather_repo_info(self, repo_dir: Path) -> dict[str, Any]:
        """Gather repository information and statistics."""
        info = {
            "name": repo_dir.name,
            "path": str(repo_dir),
            "size_mb": 0,
            "file_count": 0,
            "directory_count": 0,
            "languages": {},
            "largest_files": [],
        }

        try:
            # Basic directory stats
            total_size = 0
            file_count = 0
            dir_count = 0
            language_stats: dict[str, int] = {}
            large_files = []

            for root, dirs, files in os.walk(repo_dir):
                dir_count += len(dirs)

                for file in files:
                    file_path = Path(root) / file
                    try:
                        stat = file_path.stat()
                        file_size = stat.st_size
                        total_size += file_size
                        file_count += 1

                        # Track languages by extension
                        ext = file_path.suffix.lower()
                        if ext:
                            language_stats[ext] = language_stats.get(ext, 0) + 1

                        # Track large files
                        if file_size > 100 * 1024:  # Files > 100KB
                            large_files.append(
                                {
                                    "path": str(file_path.relative_to(repo_dir)),
                                    "size_kb": file_size / 1024,
                                }
                            )

                    except (OSError, PermissionError):
                        continue

            # Sort and limit large files
            large_files.sort(
                key=lambda x: float(str(x.get("size_kb", 0))), reverse=True
            )
            large_files = large_files[:10]  # Top 10 largest files

            info.update(
                {
                    "size_mb": total_size / (1024 * 1024),
                    "file_count": file_count,
                    "directory_count": dir_count,
                    "languages": dict(
                        sorted(
                            language_stats.items(), key=lambda x: x[1], reverse=True
                        )[:20]
                    ),
                    "largest_files": large_files,
                }
            )

            # Look for common repository files
            common_files = [
                "README.md",
                "README.rst",
                "README.txt",
                "LICENSE",
                "package.json",
                "requirements.txt",
                "Cargo.toml",
                "go.mod",
            ]
            found_files = []

            for common_file in common_files:
                file_path = repo_dir / common_file
                if file_path.exists():
                    found_files.append(common_file)

            info["common_files"] = found_files

        except Exception as e:
            self.logger.warning(f"Error gathering repository info: {e}")

        return info
