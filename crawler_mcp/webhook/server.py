#!/usr/bin/env python3
"""
GitHub Organization Webhook Server for Crawler MCP
Processes PR comments and reviews from GitHub webhooks to extract AI prompts.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from rich.console import Console
from rich.logging import RichHandler

from ..clients.qdrant_http_client import QdrantClient

# Import existing embedding infrastructure (reuse existing code)
from ..clients.tei_client import TEIEmbeddingsClient
from ..crawl_core.batch_utils import pack_items_into_batches
from ..optimized_config import OptimizedConfig

# Import OutputManager for PR CLI integration
from ..utils.output_manager import OutputManager

# Load environment variables from multiple known locations
load_dotenv()  # current working directory
package_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=package_env, override=False)
project_env = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=project_env, override=False)
# Legacy optimized env path
optimized_env_path = (
    Path(__file__).resolve().parents[1] / "crawlers" / "optimized" / ".env"
)
load_dotenv(dotenv_path=optimized_env_path, override=False)

# Configure Rich logging to match MCP server
console = Console()
rich_handler = RichHandler(
    console=console,
    show_path=False,
    show_time=True,
    rich_tracebacks=True,
    markup=True,
    log_time_format="[%H:%M:%S]",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich_handler],
)

# Add file logging similar to MCP server
log_path = Path("logs/webhook.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
)
logging.getLogger().addHandler(file_handler)

# Disable uvicorn logger propagation to prevent duplicate messages
logging.getLogger("uvicorn").propagate = False
logging.getLogger("uvicorn.error").propagate = False
logging.getLogger("uvicorn.access").propagate = False

logger = logging.getLogger(__name__)


def sanitize_url_for_logging(url: str) -> str:
    """
    Sanitize URL for logging by removing sensitive query parameters and credentials.

    Args:
        url: The URL to sanitize

    Returns:
        Sanitized URL safe for logging
    """
    try:
        parsed = urlparse(url)
        # Remove userinfo (username:password) and query parameters that might contain sensitive data
        sanitized = parsed._replace(
            netloc=parsed.netloc.split("@")[-1]
            if "@" in parsed.netloc
            else parsed.netloc,
            query="",  # Remove all query parameters to be safe
            fragment="",  # Remove fragment as well
        )
        return urlunparse(sanitized)
    except Exception:
        return "[URL parsing failed]"


class WebhookConfig:
    """Configuration for the webhook server."""

    def __init__(self) -> None:
        self.github_webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.repos_to_track = os.getenv(
            "REPOS_TO_TRACK", "*"
        )  # '*' for all, or comma-separated list
        self.script_path = os.getenv(
            "WEBHOOK_SCRIPT_PATH", "./scripts/extract_coderabbit_prompts.py"
        )
        self.output_dir = os.getenv("WEBHOOK_OUTPUT_DIR", "./webhook_outputs")
        self.max_concurrent_processes = int(
            os.getenv("WEBHOOK_MAX_CONCURRENT_PROCESSES", "5")
        )

        # New PR CLI integration settings
        self.use_pr_cli = os.getenv("WEBHOOK_USE_PR_CLI", "true").lower() == "true"
        self.pr_filters = os.getenv("WEBHOOK_PR_FILTERS", "{}")
        self.preserve_markdown = (
            os.getenv("WEBHOOK_PRESERVE_MARKDOWN", "false").lower() == "true"
        )

        # PR Embedding configuration (reuses existing infrastructure)
        self.enable_pr_embeddings = (
            os.getenv("WEBHOOK_ENABLE_PR_EMBEDDINGS", "true").lower() == "true"
        )

        # Event filtering
        self.process_reviews = os.getenv("PROCESS_REVIEWS", "true").lower() == "true"
        self.process_review_comments = (
            os.getenv("PROCESS_REVIEW_COMMENTS", "true").lower() == "true"
        )
        self.process_issue_comments = (
            os.getenv("PROCESS_ISSUE_COMMENTS", "true").lower() == "true"
        )

        # Bot filtering
        self.bot_patterns = [
            pattern.strip()
            for pattern in os.getenv(
                "BOT_PATTERNS",
                "coderabbitai[bot],copilot-pull-request-reviewer[bot],Copilot",
            ).split(",")
        ]

        # Enhanced filtering options (from old script patterns)
        self.filter_dismissed_reviews = (
            os.getenv("WEBHOOK_FILTER_DISMISSED_REVIEWS", "true").lower() == "true"
        )
        self.filter_resolved_threads = (
            os.getenv("WEBHOOK_FILTER_RESOLVED_THREADS", "true").lower() == "true"
        )
        self.filter_stale_code = (
            os.getenv("WEBHOOK_FILTER_STALE_CODE", "true").lower() == "true"
        )
        self.filter_old_comments = (
            os.getenv("WEBHOOK_FILTER_OLD_COMMENTS", "true").lower() == "true"
        )
        self.filter_pre_force_push = (
            os.getenv("WEBHOOK_FILTER_PRE_FORCE_PUSH", "true").lower() == "true"
        )
        self.max_comment_age_days = int(os.getenv("WEBHOOK_MAX_COMMENT_AGE_DAYS", "30"))
        self.max_update_age_days = int(os.getenv("WEBHOOK_MAX_UPDATE_AGE_DAYS", "7"))
        self.verbose_filtering = (
            os.getenv("WEBHOOK_VERBOSE_FILTERING", "false").lower() == "true"
        )

        self.validate()

        # Initialize OutputManager if using PR CLI
        self._output_manager = None
        if self.use_pr_cli:
            self._output_manager = OutputManager(base_dir=self.output_dir)

    @property
    def output_manager(self) -> OutputManager:
        """Get OutputManager instance."""
        if self._output_manager is None:
            self._output_manager = OutputManager(base_dir=self.output_dir)
        return self._output_manager

    def validate(self) -> None:
        """Validate configuration."""
        if not self.github_webhook_secret:
            raise ValueError("GITHUB_WEBHOOK_SECRET is required")
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN is required")
        if not Path(self.script_path).exists():
            logger.warning(f"Script path {self.script_path} does not exist")


class WebhookProcessor:
    """Handles webhook event processing."""

    def __init__(self, config: WebhookConfig):
        self.config = config
        self.active_processes: dict[str, Any] = {}
        self.process_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.stats = {
            "total_webhooks": 0,
            "processed_events": 0,
            "failed_events": 0,
            "active_processes": 0,
        }

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize embedding infrastructure (reuse existing code)
        self.opt_config = OptimizedConfig.from_env()
        self.tei_client = None
        self.qdrant_client = None

        # Only initialize if embeddings are enabled in both configs
        if self.config.enable_pr_embeddings and self.opt_config.enable_embeddings:
            try:
                self.tei_client = TEIEmbeddingsClient(
                    base_url=self.opt_config.tei_endpoint, timeout_s=30.0
                )
                logger.info(f"Initialized TEI client: {self.opt_config.tei_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to initialize TEI client: {e}")

        if self.config.enable_pr_embeddings and self.opt_config.enable_qdrant:
            try:
                self.qdrant_client = QdrantClient(
                    base_url=self.opt_config.qdrant_url,
                    api_key=self.opt_config.qdrant_api_key,
                    timeout_s=30.0,
                )
                logger.info(
                    f"Initialized Qdrant client: {sanitize_url_for_logging(self.opt_config.qdrant_url)}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant client: {e}")

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify GitHub webhook signature."""
        if not signature:
            return False

        expected = (
            "sha256="
            + hmac.new(
                self.config.github_webhook_secret.encode(), payload, hashlib.sha256
            ).hexdigest()
        )

        return hmac.compare_digest(expected, signature)

    def should_process_repo(self, repo_name: str) -> bool:
        """Check if repository should be processed."""
        if self.config.repos_to_track == "*":
            return True
        return repo_name in self.config.repos_to_track.split(",")

    def should_process_event(self, event_type: str, payload: dict[str, Any]) -> bool:
        """Check if event should be processed."""
        # Filter by event type
        event_filters = {
            "pull_request_review": self.config.process_reviews,
            "pull_request_review_comment": self.config.process_review_comments,
            "issue_comment": self.config.process_issue_comments,
        }

        if not event_filters.get(event_type, False):
            return False

        # Check if it's a PR comment (not issue comment)
        if event_type == "issue_comment" and not payload.get("issue", {}).get(
            "pull_request"
        ):
            return False

        # Filter by action
        action = payload.get("action", "")
        return action in ["created", "edited", "submitted"]

    def is_relevant_comment(self, comment_body: str, author: str) -> bool:
        """Check if comment contains relevant content using enhanced extraction logic."""
        if not comment_body:
            return False

        # Import here to avoid circular imports
        from ..utils.content_extractor import (
            extract_all_content,
            is_content_relevant,
        )

        # Use the enhanced content relevance check
        if is_content_relevant(comment_body, author):
            # Additional check: ensure we actually have extractable content
            extracted = extract_all_content(comment_body, author)
            return len(extracted) > 0

        return False

    async def process_webhook_event(
        self, event_type: str, payload: dict[str, Any]
    ) -> None:
        """Process a webhook event."""
        try:
            repo = payload.get("repository", {}).get("full_name", "")
            if not repo:
                logger.warning("No repository found in payload")
                return

            if not self.should_process_repo(repo):
                logger.info(f"Repository {repo} not in tracking list")
                return

            if not self.should_process_event(event_type, payload):
                logger.info(f"Event {event_type} not processed for {repo}")
                return

            # Extract PR number
            pr_number = None
            if event_type == "pull_request_review":
                pr_number = payload.get("pull_request", {}).get("number")
            elif event_type in ["pull_request_review_comment", "issue_comment"]:
                pr_number = payload.get("pull_request", {}).get(
                    "number"
                ) or payload.get("issue", {}).get("number")

            if not pr_number:
                logger.warning(f"No PR number found for {event_type} in {repo}")
                return

            # Check if comment is relevant
            comment_body = ""
            comment_author = ""

            if "comment" in payload:
                comment_body = payload["comment"].get("body", "")
                comment_author = payload["comment"].get("user", {}).get("login", "")
            elif "review" in payload:
                comment_body = payload["review"].get("body", "")
                comment_author = payload["review"].get("user", {}).get("login", "")

            if not self.is_relevant_comment(comment_body, comment_author):
                logger.info(f"Comment not relevant for {repo}#{pr_number}")
                return

            # Queue processing
            await self.queue_extraction(repo, pr_number, event_type)
            self.stats["processed_events"] += 1

        except Exception as e:
            logger.error(f"Error processing webhook event: {e}")
            self.stats["failed_events"] += 1

    async def queue_extraction(
        self, repo: str, pr_number: int, event_type: str
    ) -> None:
        """Queue extraction task."""
        task_id = f"{repo}#{pr_number}"

        # Check if already processing this PR
        if task_id in self.active_processes:
            logger.info(f"Already processing {task_id}, skipping")
            return

        await self.process_queue.put(
            {
                "repo": repo,
                "pr_number": pr_number,
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"Queued extraction for {task_id}")

    async def run_extraction_script(self, repo: str, pr_number: int) -> None:
        """Run extraction using either the old script or new PR CLI."""
        task_id = f"{repo}#{pr_number}"

        try:
            self.active_processes[task_id] = datetime.now()
            self.stats["active_processes"] = len(self.active_processes)

            owner, name = repo.split("/")

            if self.config.use_pr_cli:
                await self._run_pr_cli_extraction(owner, name, pr_number, task_id)
            else:
                await self._run_legacy_script_extraction(
                    owner, name, pr_number, task_id
                )

        except Exception as e:
            logger.error(f"Error running extraction for {task_id}: {e}")
        finally:
            if task_id in self.active_processes:
                del self.active_processes[task_id]
            self.stats["active_processes"] = len(self.active_processes)

    async def _run_pr_cli_extraction(
        self, owner: str, name: str, pr_number: int, task_id: str
    ) -> None:
        """Run extraction using the new PR CLI tool."""
        # Prepare PR CLI command
        cmd = [
            sys.executable,
            "-m",
            "crawler_mcp.cli.pr",
            "pr-list",
            "--owner",
            owner,
            "--repo",
            name,
            "--pr",
            str(pr_number),
            "--ndjson",  # Use structured output
        ]

        # Skip bot filtering - capture all comments regardless of author
        # if self.config.bot_patterns:
        #     cmd.extend(["--bots", ",".join(self.config.bot_patterns)])

        # Apply custom PR filters from environment
        try:
            pr_filters = json.loads(self.config.pr_filters)
            if pr_filters.get("only_unresolved"):
                cmd.append("--only-unresolved")
            if "min_length" in pr_filters:
                cmd.extend(["--min-length", str(pr_filters["min_length"])])

            # Add enhanced filtering options
            pr_filters.update(
                {
                    "filter_dismissed_reviews": self.config.filter_dismissed_reviews,
                    "filter_resolved_threads": self.config.filter_resolved_threads,
                    "filter_stale_code": self.config.filter_stale_code,
                    "filter_old_comments": self.config.filter_old_comments,
                    "filter_pre_force_push": self.config.filter_pre_force_push,
                    "max_comment_age_days": self.config.max_comment_age_days,
                    "max_update_age_days": self.config.max_update_age_days,
                }
            )

            # Add filters as JSON argument for the CLI
            cmd.extend(["--filters", json.dumps(pr_filters)])

        except (json.JSONDecodeError, KeyError):
            # Still set enhanced filtering options with defaults
            enhanced_filters = {
                "filter_dismissed_reviews": self.config.filter_dismissed_reviews,
                "filter_resolved_threads": self.config.filter_resolved_threads,
                "filter_stale_code": self.config.filter_stale_code,
                "filter_old_comments": self.config.filter_old_comments,
                "filter_pre_force_push": self.config.filter_pre_force_push,
                "max_comment_age_days": self.config.max_comment_age_days,
                "max_update_age_days": self.config.max_update_age_days,
            }
            cmd.extend(["--filters", json.dumps(enhanced_filters)])

        # Set environment variables for subprocess
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = self.config.github_token

        # Run PR CLI extraction
        logger.info(f"Running PR CLI extraction for {task_id}: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            cwd=Path.cwd(),  # Run from project root
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info(f"PR CLI extraction completed for {task_id}")
            if stdout:
                logger.debug(f"Output: {stdout.decode()}")

            # Optionally convert to markdown for backward compatibility
            if self.config.preserve_markdown:
                await self._convert_to_markdown(owner, name, pr_number)

            # Generate embeddings using existing infrastructure
            if self.config.enable_pr_embeddings:
                try:
                    embed_result = await self._embed_pr_items(owner, name, pr_number)
                    logger.info(f"[PR-EMBED] Embedding result: {embed_result}")
                except Exception as e:
                    logger.error(
                        f"[PR-EMBED] Failed to embed PR items for {task_id}: {e}"
                    )
        else:
            logger.error(f"PR CLI extraction failed for {task_id}: {stderr.decode()}")

    async def _run_legacy_script_extraction(
        self, owner: str, name: str, pr_number: int, task_id: str
    ) -> None:
        """Run extraction using the legacy script."""
        # Prepare legacy script command (resolve script path to absolute)
        script_path = str(Path(self.config.script_path).resolve())
        cmd = ["python", script_path, owner, name, str(pr_number)]

        # Set environment variables for subprocess
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = self.config.github_token

        # Run legacy extraction script
        logger.info(f"Running legacy extraction for {task_id}: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            cwd=self.config.output_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info(f"Legacy extraction completed for {task_id}")
            if stdout:
                logger.debug(f"Output: {stdout.decode()}")
        else:
            logger.error(f"Legacy extraction failed for {task_id}: {stderr.decode()}")

    async def _convert_to_markdown(self, owner: str, repo: str, pr_number: int) -> None:
        """Convert PR CLI JSON output to markdown format for backward compatibility."""
        try:
            paths = self.config.output_manager.get_pr_output_paths(
                owner, repo, pr_number
            )

            # Check if items file exists
            if paths["items"].exists():
                # Read NDJSON items
                items = []
                with open(paths["items"], encoding="utf-8") as f:
                    for line in f:
                        items.append(json.loads(line.strip()))

                # Generate markdown content (currently not persisted)
                _md_content = self._generate_markdown_from_items(
                    items, owner, repo, pr_number
                )

        except Exception as e:
            logger.warning(
                f"Failed to convert to markdown for {owner}/{repo}#{pr_number}: {e}"
            )

    def _generate_markdown_from_items(
        self, items: list[dict], owner: str, repo: str, pr_number: int
    ) -> str:
        """Generate markdown content from PR CLI items."""
        content_lines = [
            f"# AI Review Content from PR #{pr_number}",
            "",
            f"**Extracted from PR:** https://github.com/{owner}/{repo}/pull/{pr_number}",
            f"**Items found:** {len(items)}",
            "**Generated by:** Crawler MCP Webhook Server (PR CLI)",
            "",
            "---",
            "",
        ]

        for item in items:
            item_type = item.get("item_type", "unknown")
            author = item.get("author", "unknown")
            path = item.get("path", "")
            line = item.get("line", "")
            body = item.get("body", "")

            # Format file reference
            file_ref = ""
            if path and line:
                file_ref = f" - {path}:{line}"
            elif path:
                file_ref = f" - {path}"

            # Add item to markdown
            content_lines.extend(
                [
                    f"- [ ] [{item_type.upper()} - {author}{file_ref}]",
                    f"{body}",
                    "",
                    "---",
                    "",
                ]
            )

        return "\n".join(content_lines)

    async def process_queue_worker(self) -> None:
        """Background worker to process the extraction queue."""
        while True:
            try:
                # Wait for task
                task = await self.process_queue.get()

                # Check concurrent limit
                if len(self.active_processes) >= self.config.max_concurrent_processes:
                    # Put task back and wait
                    await self.process_queue.put(task)
                    await asyncio.sleep(1)
                    continue

                # Process task
                await self.run_extraction_script(task["repo"], task["pr_number"])

                # Mark task as done
                self.process_queue.task_done()

            except Exception as e:
                logger.error(f"Error in queue worker: {e}")
                await asyncio.sleep(5)

    async def _embed_pr_items(
        self, owner: str, repo: str, pr_number: int
    ) -> dict[str, Any]:
        """Generate embeddings for PR items using existing TEI/Qdrant infrastructure."""

        # Check if embedding is enabled and clients are available
        if not (self.tei_client and self.qdrant_client):
            return {
                "status": "skipped",
                "reason": "embeddings_disabled",
                "embedded_count": 0,
            }

        task_id = f"{repo}#{pr_number}"
        start_time = time.time()
        logger.info(
            f"[PR-EMBED] 🚀 Starting embedding for {task_id} using existing infrastructure"
        )

        try:
            # Read items from NDJSON using existing output manager
            items_path = self.config.output_manager.get_pr_output_paths(
                owner, repo, pr_number
            )["items"]

            if not items_path.exists():
                logger.warning(f"[PR-EMBED] Items file not found: {items_path}")
                return {
                    "status": "error",
                    "reason": "items_file_not_found",
                    "embedded_count": 0,
                }

            # Load items from NDJSON
            items = []
            with open(items_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line.strip()))

            # Filter items that have text content to embed
            embeddable_items = [item for item in items if item.get("body", "").strip()]

            if not embeddable_items:
                logger.info(f"[PR-EMBED] No embeddable content found for {task_id}")
                return {
                    "status": "skipped",
                    "reason": "no_content",
                    "embedded_count": 0,
                }

            logger.info(
                f"[PR-EMBED] Processing {len(embeddable_items)}/{len(items)} items with content"
            )

            # Use shared batching utility to avoid TEI payload limits
            logger.info(
                f"[PR-EMBED] Generating embeddings via TEI: {self.opt_config.tei_endpoint}"
            )

            # Create batches using existing algorithm from optimized crawler
            batches = pack_items_into_batches(
                embeddable_items,
                text_extractor=lambda item: item["body"],
                target_chars=self.opt_config.tei_target_chars_per_batch or 64000,
                max_items=self.opt_config.tei_max_client_batch_size or 128,
                parallel_workers=self.opt_config.tei_parallel_requests or 4,
            )

            logger.info(
                f"[PR-EMBED] Processing {len(embeddable_items)} items in {len(batches)} batches"
            )

            # Process batches in parallel using existing patterns
            async with self.tei_client as tei:
                sem = asyncio.Semaphore(self.opt_config.tei_parallel_requests or 4)
                tasks = []

                async def process_batch(batch_items):
                    """Process a single batch of items."""
                    async with sem:
                        batch_texts = [item["body"] for _, item in batch_items]
                        batch_embeddings = await tei.embed_texts(batch_texts)
                        return [
                            (item, embedding)
                            for (_, item), embedding in zip(
                                batch_items, batch_embeddings, strict=False
                            )
                        ]

                # Launch all batches in parallel
                for batch in batches:
                    tasks.append(asyncio.create_task(process_batch(batch)))

                # Collect results as they complete
                item_embedding_pairs = []
                for task in asyncio.as_completed(tasks):
                    batch_results = await task
                    item_embedding_pairs.extend(batch_results)

            if len(item_embedding_pairs) != len(embeddable_items):
                raise ValueError(
                    f"Embedding count mismatch: {len(item_embedding_pairs)} vs {len(embeddable_items)}"
                )

            logger.info(
                f"[PR-EMBED] Successfully generated embeddings for all {len(item_embedding_pairs)} items"
            )

            # Prepare points for Qdrant using existing client patterns
            points = []
            for item, embedding in item_embedding_pairs:
                # Generate unique UUID for this PR item using deterministic hashing
                content_hash = hashlib.sha256(item["body"].encode("utf-8")).hexdigest()
                item_identifier = f"pr-{owner}-{repo}-{pr_number}-{item.get('item_id', content_hash[:16])}"
                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, item_identifier))

                # Create point with comprehensive metadata
                point = {
                    "id": point_uuid,
                    "vector": embedding,
                    "payload": {
                        # Core PR metadata
                        "source_type": "github_pr",
                        "owner": owner,
                        "repo": repo,
                        "pr_number": pr_number,
                        "item_id": str(item.get("item_id", "")),
                        "item_type": item.get("item_type", ""),
                        "author": item.get("author", ""),
                        "created_at": item.get("created_at", ""),
                        "canonical_url": item.get("canonical_url", ""),
                        # Identifiers
                        "original_id": item_identifier,  # Store original string identifier
                        "content_hash": content_hash,
                        # Content metadata
                        "body": item["body"],  # Store full text for retrieval
                        "word_count": len(item["body"].split()),
                        # File/code context (if available)
                        "file_path": item.get("path", ""),
                        "line_number": item.get("line", ""),
                        "commit_sha": item.get("commit_id", ""),
                        "original_commit_sha": item.get("original_commit_id", ""),
                        # Embedding metadata
                        "embedded_at": datetime.now().isoformat(),
                        "embedding_model": self.opt_config.tei_model_name,
                    },
                }
                points.append(point)

            # Store embeddings using existing Qdrant client (upsert will overwrite by ID)
            collection = self.opt_config.qdrant_collection
            logger.info(
                f"[PR-EMBED] Storing {len(points)} vectors in Qdrant collection: {collection}"
            )

            # Upsert vectors (will overwrite existing ones with same IDs)
            async with self.qdrant_client as qdrant:
                await qdrant.upsert(collection, points)

            duration = time.time() - start_time
            logger.info(f"[PR-EMBED] ✅ Successfully embedded {task_id}:")
            logger.info(f"[PR-EMBED]   • Vectors stored: {len(points)}")
            logger.info(f"[PR-EMBED]   • Time taken: {duration:.2f}s")
            logger.info(f"[PR-EMBED]   • Avg per item: {duration / len(points):.3f}s")

            return {
                "status": "success",
                "embedded_count": len(points),
                "total_items": len(items),
                "task_id": task_id,
                "duration": duration,
                "collection": collection,
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"[PR-EMBED] ❌ Failed to embed {task_id} after {duration:.2f}s: {e}"
            )
            return {
                "status": "error",
                "reason": str(e),
                "embedded_count": 0,
                "task_id": task_id,
            }


# Global instances
config = WebhookConfig()
processor = WebhookProcessor(config)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    # Get configuration
    host = "0.0.0.0"
    port = int(os.getenv("WEBHOOK_PORT", "38080"))

    # Start background worker
    worker_task = asyncio.create_task(processor.process_queue_worker())

    # Announce startup with host:port info like MCP server
    logger.info("")
    logger.info("[bold green]✨ Webhook server ready on %s:%s[/bold green]", host, port)
    logger.info("  • PR CLI enabled: %s", config.use_pr_cli)
    logger.info("  • Output directory: %s", config.pr_output_base_dir)
    logger.info("  • Bot patterns: %s", ", ".join(config.bot_patterns))
    logger.info("")

    yield

    # Cleanup
    worker_task.cancel()
    logger.info("Webhook server stopped")


# Create FastAPI app
app = FastAPI(
    title="GitHub Webhook Processor",
    description="Processes GitHub organization webhooks for AI prompt extraction",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/webhook")
async def handle_webhook(
    request: Request, background_tasks: BackgroundTasks
) -> JSONResponse:
    """Handle GitHub webhook events."""
    try:
        # Get headers
        signature = request.headers.get("X-Hub-Signature-256", "")
        event_type = request.headers.get("X-GitHub-Event", "")
        delivery_id = request.headers.get("X-GitHub-Delivery", "")

        # Read payload
        payload_bytes = await request.body()

        # Verify signature
        if not processor.verify_signature(payload_bytes, signature):
            logger.warning(f"Invalid signature for delivery {delivery_id}")
            raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse JSON
        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="Invalid JSON") from e

        processor.stats["total_webhooks"] += 1

        logger.info(f"Received {event_type} webhook for delivery {delivery_id}")

        # Process in background
        background_tasks.add_task(processor.process_webhook_event, event_type, payload)

        return JSONResponse(
            {"status": "accepted", "delivery_id": delivery_id, "event_type": event_type}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "stats": processor.stats,
        "queue_size": processor.process_queue.qsize(),
        "active_processes": len(processor.active_processes),
        "pr_cli_enabled": config.use_pr_cli,
    }

    # Add PR CLI specific health info
    if config.use_pr_cli:
        try:
            output_mgr = config.output_manager
            pr_count = 0
            if output_mgr.pr_dir.exists():
                pr_count = len(
                    [
                        d
                        for d in output_mgr.pr_dir.iterdir()
                        if d.is_dir() and not d.name.startswith("_")
                    ]
                )

            health_info["pr_outputs"] = {
                "total_prs": pr_count,
                "output_dir": str(output_mgr.pr_dir),
                "output_size_mb": round(output_mgr.get_total_size() / (1024 * 1024), 2),
            }
        except Exception as e:
            health_info["pr_outputs"] = {"error": str(e)}

    return JSONResponse(health_info)


@app.get("/stats")
async def get_stats() -> JSONResponse:
    """Get processing statistics."""
    stats_info = {
        "stats": processor.stats,
        "queue_size": processor.process_queue.qsize(),
        "active_processes": list(processor.active_processes.keys()),
        "config": {
            "repos_tracked": config.repos_to_track,
            "max_concurrent": config.max_concurrent_processes,
            "bot_patterns": config.bot_patterns,
            "pr_cli_enabled": config.use_pr_cli,
            "output_mode": "pr_cli" if config.use_pr_cli else "legacy",
        },
    }

    # Add PR CLI specific stats
    if config.use_pr_cli:
        try:
            stats_info["pr_cli_config"] = {
                "output_base_dir": config.pr_output_base_dir,
                "preserve_markdown": config.preserve_markdown,
                "filters": config.pr_filters,
            }

            output_mgr = config.output_manager
            if output_mgr.pr_dir.exists():
                recent_outputs = []
                for pr_folder in sorted(
                    output_mgr.pr_dir.iterdir(),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[:5]:
                    if pr_folder.is_dir() and not pr_folder.name.startswith("_"):
                        recent_outputs.append(
                            {
                                "folder": pr_folder.name,
                                "modified": pr_folder.stat().st_mtime,
                                "size_kb": round(
                                    sum(
                                        f.stat().st_size
                                        for f in pr_folder.rglob("*")
                                        if f.is_file()
                                    )
                                    / 1024,
                                    2,
                                ),
                            }
                        )

                stats_info["recent_pr_outputs"] = recent_outputs
        except Exception as e:
            stats_info["pr_cli_error"] = str(e)

    return JSONResponse(stats_info)


async def get_recent_prs_for_user(
    github_token: str, days: int = 7
) -> list[dict[str, Any]]:
    """Get recent PRs from all accessible repositories."""
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    recent_prs = []
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

    async with httpx.AsyncClient() as client:
        try:
            # Get all repositories the user has access to
            repos_response = await client.get(
                "https://api.github.com/user/repos",
                headers=headers,
                params={"per_page": 100, "sort": "updated", "direction": "desc"},
            )
            repos_response.raise_for_status()
            repos = repos_response.json()

            for repo in repos:
                repo_name = repo["full_name"]

                try:
                    # Get recent PRs for this repo
                    prs_response = await client.get(
                        f"https://api.github.com/repos/{repo_name}/pulls",
                        headers=headers,
                        params={
                            "state": "all",
                            "sort": "updated",
                            "direction": "desc",
                            "per_page": 5,
                        },
                    )
                    prs_response.raise_for_status()
                    prs = prs_response.json()

                    # Filter PRs by date and add to results
                    for pr in prs:
                        updated_at = datetime.fromisoformat(
                            pr["updated_at"].replace("Z", "+00:00")
                        )
                        if updated_at >= datetime.fromisoformat(
                            since.replace("Z", "+00:00")
                        ):
                            recent_prs.append(
                                {
                                    "repo": repo_name,
                                    "pr_number": pr["number"],
                                    "title": pr["title"],
                                    "state": pr["state"],
                                    "updated_at": pr["updated_at"],
                                    "url": pr["html_url"],
                                }
                            )

                except httpx.HTTPStatusError as e:
                    logger.warning(f"Could not fetch PRs for {repo_name}: {e}")
                    continue

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch repositories: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to fetch repositories"
            ) from e

    # Sort by updated date descending
    recent_prs.sort(key=lambda x: x["updated_at"], reverse=True)
    return recent_prs


@app.get("/recent")
async def get_recent_prs(days: int = 7) -> JSONResponse:
    """Get recent PRs from all accessible repositories.

    Query parameters:
    - days: Number of days to look back (default: 7)
    """
    try:
        if not config.github_token:
            raise HTTPException(status_code=500, detail="GitHub token not configured")

        recent_prs = await get_recent_prs_for_user(config.github_token, days)

        return JSONResponse(
            {
                "recent_prs": recent_prs,
                "total_prs": len(recent_prs),
                "days_back": days,
                "message": f"Found {len(recent_prs)} PRs updated in the last {days} days",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching recent PRs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/batch")
async def batch_extraction(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Trigger batch extraction for multiple PRs.

    Expected JSON payload:
    {
        "prs": [
            {"owner": "user", "repo": "repo1", "pr_number": 123},
            {"owner": "user", "repo": "repo2", "pr_number": 456}
        ],
        "auto_discover": false,  // Optional: auto-discover recent PRs
        "days": 7               // Optional: days to look back for auto-discover
    }
    """
    try:
        # Parse JSON payload
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid JSON") from e

        prs_to_process = []

        # Check if auto-discovery is requested
        if payload.get("auto_discover", False):
            if not config.github_token:
                raise HTTPException(
                    status_code=500,
                    detail="GitHub token not configured for auto-discovery",
                )

            days = payload.get("days", 7)
            recent_prs = await get_recent_prs_for_user(config.github_token, days)

            for pr in recent_prs:
                owner, repo_name = pr["repo"].split("/", 1)
                prs_to_process.append(
                    {"owner": owner, "repo": repo_name, "pr_number": pr["pr_number"]}
                )
        else:
            # Use provided PR list
            prs_list = payload.get("prs", [])
            if not prs_list:
                raise HTTPException(
                    status_code=400,
                    detail="Either provide 'prs' list or set 'auto_discover': true",
                )

            # Validate each PR entry
            for pr in prs_list:
                if not all(key in pr for key in ["owner", "repo", "pr_number"]):
                    raise HTTPException(
                        status_code=400,
                        detail="Each PR must have 'owner', 'repo', and 'pr_number'",
                    )
                prs_to_process.append(pr)

        # Queue all extractions
        queued_tasks = []
        for pr in prs_to_process:
            try:
                pr_number = int(pr["pr_number"])
                repo = f"{pr['owner']}/{pr['repo']}"

                await processor.queue_extraction(repo, pr_number, "batch")
                queued_tasks.append(
                    {"repo": repo, "pr_number": pr_number, "status": "queued"}
                )

                logger.info(f"Batch extraction queued for {repo}#{pr_number}")

            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid PR number for {pr}: {e}")
                queued_tasks.append(
                    {
                        "repo": f"{pr['owner']}/{pr['repo']}",
                        "pr_number": pr["pr_number"],
                        "status": "error",
                        "error": "Invalid PR number",
                    }
                )

        return JSONResponse(
            {
                "status": "batch_queued",
                "total_prs": len(prs_to_process),
                "successfully_queued": sum(
                    1 for task in queued_tasks if task["status"] == "queued"
                ),
                "failed": sum(1 for task in queued_tasks if task["status"] == "error"),
                "tasks": queued_tasks,
                "message": f"Queued extraction for {len(queued_tasks)} PRs",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling batch extraction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/manual")
async def manual_extraction(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Manually trigger extraction for a specific PR.

    Expected JSON payload:
    {
        "owner": "github-username",
        "repo": "repository-name",
        "pr_number": 123
    }
    """
    try:
        # Parse JSON payload
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid JSON") from e

        # Validate required fields
        owner = payload.get("owner")
        repo_name = payload.get("repo")
        pr_number = payload.get("pr_number")

        if not all([owner, repo_name, pr_number]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: owner, repo, pr_number",
            )

        # Validate pr_number is an integer
        try:
            pr_number = int(pr_number)
        except (TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400, detail="pr_number must be an integer"
            ) from e

        # Construct repo full name
        repo = f"{owner}/{repo_name}"

        logger.info(f"Manual extraction requested for {repo}#{pr_number}")

        # Queue the extraction
        await processor.queue_extraction(repo, pr_number, "manual")

        return JSONResponse(
            {
                "status": "queued",
                "repo": repo,
                "pr_number": pr_number,
                "message": f"Extraction queued for {repo}#{pr_number}",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling manual extraction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/apply-suggestions")
async def apply_pr_suggestions(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Apply code suggestions from PR comments using PR CLI.

    Expected JSON payload:
    {
        "owner": "github-username",
        "repo": "repository-name",
        "pr_number": 123,
        "strategy": "dry-run",  // or "commit"
        "only_unresolved": true
    }
    """
    try:
        if not config.use_pr_cli:
            raise HTTPException(
                status_code=400,
                detail="PR CLI not enabled. Set WEBHOOK_USE_PR_CLI=true",
            )

        payload = await request.json()
        owner = payload.get("owner")
        repo_name = payload.get("repo")
        pr_number = payload.get("pr_number")
        strategy = payload.get("strategy", "dry-run")
        only_unresolved = payload.get("only_unresolved", False)

        if not all([owner, repo_name, pr_number]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: owner, repo, pr_number",
            )

        pr_number = int(pr_number)

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "crawler_mcp.cli.pr",
            "pr-apply-suggestions",
            "--owner",
            owner,
            "--repo",
            repo_name,
            "--pr",
            str(pr_number),
            "--strategy",
            strategy,
        ]

        if only_unresolved:
            cmd.append("--only-unresolved")

        # Set environment variables
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = config.github_token

        # Execute command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            cwd=Path.cwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            try:
                result = json.loads(stdout.decode())
                return JSONResponse(result)
            except json.JSONDecodeError:
                return JSONResponse(
                    {
                        "status": "success",
                        "message": "Suggestions applied successfully",
                        "output": stdout.decode(),
                    }
                )
        else:
            raise HTTPException(
                status_code=500, detail=f"Command failed: {stderr.decode()}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying suggestions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/mark-resolved")
async def mark_pr_items_resolved(request: Request) -> JSONResponse:
    """Mark PR items as resolved using PR CLI.

    Expected JSON payload:
    {
        "owner": "github-username",
        "repo": "repository-name",
        "pr_number": 123,
        "resolved": true,
        "item_types": "pr_review_comment,issue_comment"
    }
    """
    try:
        if not config.use_pr_cli:
            raise HTTPException(
                status_code=400,
                detail="PR CLI not enabled. Set WEBHOOK_USE_PR_CLI=true",
            )

        payload = await request.json()
        owner = payload.get("owner")
        repo_name = payload.get("repo")
        pr_number = payload.get("pr_number")
        resolved = payload.get("resolved", True)
        item_types = payload.get("item_types", "")

        if not all([owner, repo_name, pr_number]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: owner, repo, pr_number",
            )

        pr_number = int(pr_number)

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "crawler_mcp.cli.pr",
            "pr-mark-resolved",
            "--owner",
            owner,
            "--repo",
            repo_name,
            "--pr",
            str(pr_number),
            "--resolved",
            "true" if resolved else "false",
        ]

        if item_types:
            cmd.extend(["--item-types", item_types])

        # Set environment variables
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = config.github_token

        # Execute command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            cwd=Path.cwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            try:
                result = json.loads(stdout.decode())
                return JSONResponse(result)
            except json.JSONDecodeError:
                return JSONResponse(
                    {
                        "status": "success",
                        "message": "Items marked as resolved",
                        "output": stdout.decode(),
                    }
                )
        else:
            raise HTTPException(
                status_code=500, detail=f"Command failed: {stderr.decode()}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking resolved: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/pr-outputs")
async def list_pr_outputs() -> JSONResponse:
    """List available PR outputs."""
    try:
        if not config.use_pr_cli:
            return JSONResponse(
                {"message": "PR CLI not enabled", "outputs": [], "legacy_mode": True}
            )

        output_mgr = config.output_manager
        pr_outputs = []

        if output_mgr.pr_dir.exists():
            for pr_folder in output_mgr.pr_dir.iterdir():
                if pr_folder.is_dir() and not pr_folder.name.startswith("_"):
                    # Parse folder name: owner_repo_pr_number
                    parts = pr_folder.name.split("_")
                    if len(parts) >= 3:
                        try:
                            pr_number = int(parts[-1])
                            repo = "_".join(
                                parts[1:-1]
                            )  # Handle repos with underscores
                            owner = parts[0]

                            # Check what files exist
                            paths = output_mgr.get_pr_output_paths(
                                owner, repo, pr_number
                            )
                            files = {}
                            for key, path in paths.items():
                                files[key] = {
                                    "exists": path.exists(),
                                    "size": path.stat().st_size if path.exists() else 0,
                                    "modified": path.stat().st_mtime
                                    if path.exists()
                                    else 0,
                                }

                            pr_outputs.append(
                                {
                                    "owner": owner,
                                    "repo": repo,
                                    "pr_number": pr_number,
                                    "folder": str(pr_folder),
                                    "files": files,
                                }
                            )
                        except (ValueError, IndexError):
                            continue  # Skip invalid folder names

        return JSONResponse(
            {"pr_outputs": pr_outputs, "total": len(pr_outputs), "pr_cli_enabled": True}
        )

    except Exception as e:
        logger.error(f"Error listing PR outputs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint."""
    endpoints = {
        "webhook": "/webhook",
        "health": "/health",
        "stats": "/stats",
        "recent": "/recent (GET)",
        "batch": "/batch (POST)",
        "manual": "/manual (POST)",
    }

    # Add PR CLI endpoints if enabled
    if config.use_pr_cli:
        endpoints.update(
            {
                "apply_suggestions": "/apply-suggestions (POST)",
                "mark_resolved": "/mark-resolved (POST)",
                "pr_outputs": "/pr-outputs (GET)",
            }
        )

    return JSONResponse(
        {
            "service": "GitHub Webhook Processor",
            "version": "1.0.0",
            "status": "running",
            "pr_cli_enabled": config.use_pr_cli,
            "endpoints": endpoints,
        }
    )


def main() -> None:
    """Main entry point for webhook server."""
    uvicorn.run(
        "crawler_mcp.webhook.server:app",
        host="0.0.0.0",
        port=int(os.getenv("WEBHOOK_PORT", "38080")),
        log_level="info",
        log_config=None,  # Use our existing Rich logging instead of uvicorn's
        access_log=False,  # Disable access logs to avoid duplication
    )


if __name__ == "__main__":
    main()
