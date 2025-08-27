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
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/webhook_server.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class WebhookConfig:
    """Configuration for the webhook server."""

    def __init__(self):
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

        self.validate()

    def validate(self):
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
        self.active_processes = {}
        self.process_queue = asyncio.Queue()
        self.stats = {
            "total_webhooks": 0,
            "processed_events": 0,
            "failed_events": 0,
            "active_processes": 0,
        }

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

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
        """Check if comment contains relevant content."""
        if not comment_body:
            return False

        # Check for bot authors
        if any(
            bot in author.lower()
            for bot in [pattern.lower() for pattern in self.config.bot_patterns]
        ):
            return True

        # Check for code blocks or suggestions
        return any(
            marker in comment_body
            for marker in [
                "```",
                "📝 Committable suggestion",
                "🤖 Prompt for AI Agents",
            ]
        )

    async def process_webhook_event(self, event_type: str, payload: dict[str, Any]):
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

    async def queue_extraction(self, repo: str, pr_number: int, event_type: str):
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

    async def run_extraction_script(self, repo: str, pr_number: int):
        """Run the extraction script."""
        task_id = f"{repo}#{pr_number}"

        try:
            self.active_processes[task_id] = datetime.now()
            self.stats["active_processes"] = len(self.active_processes)

            owner, name = repo.split("/")

            # Prepare command (resolve script path to absolute)
            script_path = str(Path(self.config.script_path).resolve())
            cmd = ["python", script_path, owner, name, str(pr_number)]

            # Set environment variables for subprocess
            env = os.environ.copy()
            env["GITHUB_TOKEN"] = self.config.github_token

            # Run extraction script
            logger.info(f"Running extraction for {task_id}: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=self.config.output_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"Extraction completed for {task_id}")
                if stdout:
                    logger.debug(f"Output: {stdout.decode()}")
            else:
                logger.error(f"Extraction failed for {task_id}: {stderr.decode()}")

        except Exception as e:
            logger.error(f"Error running extraction for {task_id}: {e}")
        finally:
            if task_id in self.active_processes:
                del self.active_processes[task_id]
            self.stats["active_processes"] = len(self.active_processes)

    async def process_queue_worker(self):
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


# Global instances
config = WebhookConfig()
processor = WebhookProcessor(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Start background worker
    worker_task = asyncio.create_task(processor.process_queue_worker())
    logger.info("Webhook server started")

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
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
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
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "stats": processor.stats,
            "queue_size": processor.process_queue.qsize(),
            "active_processes": len(processor.active_processes),
        }
    )


@app.get("/stats")
async def get_stats():
    """Get processing statistics."""
    return JSONResponse(
        {
            "stats": processor.stats,
            "queue_size": processor.process_queue.qsize(),
            "active_processes": list(processor.active_processes.keys()),
            "config": {
                "repos_tracked": config.repos_to_track,
                "max_concurrent": config.max_concurrent_processes,
                "bot_patterns": config.bot_patterns,
            },
        }
    )


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
async def get_recent_prs(days: int = 7):
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
):
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
):
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


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        {
            "service": "GitHub Webhook Processor",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "webhook": "/webhook",
                "health": "/health",
                "stats": "/stats",
                "recent": "/recent (GET)",
                "batch": "/batch (POST)",
                "manual": "/manual (POST)",
            },
        }
    )


def main():
    """Main entry point for webhook server."""
    uvicorn.run(
        "crawler_mcp.webhook.server:app",
        host="0.0.0.0",
        port=int(os.getenv("WEBHOOK_PORT", "38080")),
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
