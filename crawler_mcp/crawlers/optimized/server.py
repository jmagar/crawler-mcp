"""
Optimized Crawler MCP FastMCP Server

Consolidated server that registers tools from the optimized package and
re-homes middleware and RAG tools under `crawler_mcp.crawlers.optimized`.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import atexit
import logging
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Configure logging before importing other modules
try:
    from ...config import settings
    from ...core import EmbeddingService, RagService, VectorService
    from ...core.logging import get_logger
except Exception:  # pragma: no cover - fallback for direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from crawler_mcp.config import settings  # type: ignore
    from crawler_mcp.core import (  # type: ignore
        EmbeddingService,
        RagService,
        VectorService,
    )
    from crawler_mcp.core.logging import get_logger  # type: ignore

# Tool registrations (optimized only)
try:
    from .tools.crawling import register_crawling_tools
    from .tools.github_pr_tools import register_github_pr_tools
    from .tools.rag import register_rag_tools
except Exception:  # pragma: no cover - fallback for direct execution via file path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from crawler_mcp.crawlers.optimized.tools.crawling import (
        register_crawling_tools,  # type: ignore
    )
    from crawler_mcp.crawlers.optimized.tools.github_pr_tools import (
        register_github_pr_tools,  # type: ignore
    )
    from crawler_mcp.crawlers.optimized.tools.rag import (
        register_rag_tools,  # type: ignore
    )


def setup_logging() -> None:
    """Configure rich colorized logging for the application."""
    install(show_locals=settings.debug)

    console = Console(force_terminal=True, width=120)
    rich_handler = RichHandler(
        console=console,
        show_path=settings.debug,
        show_time=True,
        rich_tracebacks=True,
        tracebacks_show_locals=settings.debug,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
    )

    if settings.log_to_file and settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("chromium").setLevel(logging.WARNING)

    logging.getLogger("uvicorn").propagate = False
    logging.getLogger("uvicorn.error").propagate = False
    logging.getLogger("uvicorn.access").propagate = False


setup_logging()
logger = get_logger(__name__)

# FastMCP instance
mcp: FastMCP = FastMCP("crawler-mcp")

# Register optimized tools only
register_github_pr_tools(mcp)
register_rag_tools(mcp)
register_crawling_tools(mcp)

logger.info("Registered optimized FastMCP tools")


@mcp.tool
async def health_check(ctx: Context, detailed: bool = False) -> dict[str, Any]:
    """Perform a health check of all services."""
    check_type = "detailed" if detailed else "lightweight"
    await ctx.info(f"Performing {check_type} health check of all services")

    try:
        health_results: dict[str, Any] = {
            "server": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            "services": {},
            "configuration": {
                "embedding_model": settings.tei_model,
                "vector_database": settings.qdrant_url,
                "embedding_service": settings.tei_url,
                "max_concurrent_crawls": settings.max_concurrent_crawls,
                "chunk_size": 1000,
                "vector_dimension": settings.qdrant_vector_size,
            },
        }
        services: dict[str, Any] = health_results["services"]

        # Embedding
        try:
            async with EmbeddingService() as embedding_service:
                embedding_healthy = await embedding_service.health_check()
                info: dict[str, Any] = {
                    "status": "healthy" if embedding_healthy else "unhealthy",
                    "url": settings.tei_url,
                    "model": settings.tei_model,
                }
                if detailed:
                    info["model_info"] = await embedding_service.get_model_info(max_size=2000)
                services["embedding"] = info
        except Exception as e:  # pragma: no cover
            services["embedding"] = {"status": "error", "error": str(e)}

        # Vector
        try:
            async with VectorService() as vector_service:
                ok = await vector_service.health_check()
                info = {
                    "status": "healthy" if ok else "unhealthy",
                    "url": settings.qdrant_url,
                    "collection": settings.qdrant_collection,
                }
                if detailed:
                    info["collection_info"] = await vector_service.get_collection_info()
                services["vector"] = info
        except Exception as e:  # pragma: no cover
            services["vector"] = {"status": "error", "error": str(e)}

        # RAG
        try:
            async with RagService() as rag_service:
                rag_health = await rag_service.health_check()
                info = {
                    "status": "healthy" if all(rag_health.values()) else "unhealthy",
                    "component_health": rag_health,
                }
                if detailed:
                    info["stats"] = await rag_service.get_stats()
                services["rag"] = info
        except Exception as e:  # pragma: no cover
            services["rag"] = {"status": "error", "error": str(e)}

        return health_results
    except Exception as e:  # pragma: no cover
        raise ToolError(f"Health check failed: {e!s}") from e


# Webhook subprocess management (unchanged)
webhook_process: subprocess.Popen[bytes] | None = None


def cleanup_webhook_server() -> None:
    global webhook_process
    if webhook_process and webhook_process.poll() is None:
        logger.info("Stopping webhook server...")
        webhook_process.terminate()
        try:
            webhook_process.wait(timeout=5)
        except Exception:
            logger.warning("Webhook server did not terminate gracefully, killing...")
            webhook_process.kill()
            webhook_process.wait()
        finally:
            webhook_process = None


def _sigterm_handler(signum: int, _frame: Any) -> None:  # pragma: no cover
    logger.info("Received signal %s, shutting down...", signum)
    cleanup_webhook_server()
    sys.exit(0)


signal.signal(signal.SIGTERM, _sigterm_handler)
atexit.register(cleanup_webhook_server)


def start_webhook_server() -> None:
    """Start the webhook server subprocess if enabled via env."""
    global webhook_process
    start_webhook = os.getenv("START_WEBHOOK_SERVER", "true").lower() == "true"
    if not start_webhook:
        logger.info("Webhook server startup disabled via START_WEBHOOK_SERVER=false")
        return

    try:
        webhook_port = os.getenv("WEBHOOK_PORT", "38080")
        logger.info("Starting webhook server on port %s", webhook_port)
        webhook_process = subprocess.Popen(
            [sys.executable, "-m", "crawler_mcp.webhook.server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        logger.info("Webhook server started with PID %s", webhook_process.pid)
    except Exception as e:
        logger.error("Failed to start webhook server: %s", e)
        webhook_process = None


async def startup_checks() -> None:
    # Placeholders for any async startup checks in the future
    return None


def main() -> None:
    try:
        console = Console()
        console.print("")
        console.print("[bold blue]🕷️  Crawler-MCP (Optimized) Server v0.1.0[/bold blue]")
        console.print("[blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/blue]")
        console.print("[dim]RAG-Enabled Web Crawling Server[/dim]")
        console.print("")

        asyncio.run(startup_checks())
        start_webhook_server()

        import uvicorn

        uvicorn.run(
            mcp.http_app(),
            host=settings.server_host,
            port=settings.server_port,
            log_level="info" if settings.debug else "warning",
            log_config=None,
            access_log=False,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        cleanup_webhook_server()
        sys.exit(0)
    except Exception as e:  # pragma: no cover
        logger.exception("Server failed to start: %s", e)
        cleanup_webhook_server()
        sys.exit(1)


if __name__ == "__main__":
    main()
