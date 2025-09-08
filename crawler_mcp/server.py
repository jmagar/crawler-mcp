"""
Crawler MCP FastMCP Server (top-level entry)

Consolidated server that registers tools from the optimized crawler package
and uses core RAG/vector services in crawler_mcp.core.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from crawler_mcp.config import settings
from crawler_mcp.core import EmbeddingService, RagService, VectorService
from crawler_mcp.core.logging import get_logger
from crawler_mcp.optimized_config import OptimizedConfig

# Tool registrations (top-level package)
from crawler_mcp.tools.crawling import register_crawling_tools
from crawler_mcp.tools.github_pr_tools import register_github_pr_tools
from crawler_mcp.tools.rag import register_rag_tools


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
            logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
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

# Workaround for FastMCP versions missing settings.mask_error_details
try:  # pragma: no cover - defensive compatibility shim
    import fastmcp.settings as _fastmcp_settings  # type: ignore

    if not hasattr(_fastmcp_settings, "mask_error_details"):
        _fastmcp_settings.mask_error_details = False  # default behavior
except Exception:  # pragma: no cover - if import fails, let FastMCP handle it
    pass

# Set environment flags after imports but before runtime initializations
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure OAuth if enabled
auth_provider = None
if settings.oauth_enabled and settings.oauth_provider == "google":
    try:
        from fastmcp.server.auth.providers.google import GoogleProvider

        auth_provider = GoogleProvider(
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            base_url=settings.google_base_url,
            required_scopes=settings.google_scopes_list,
        )
        logger.info("Google OAuth enabled with base URL: %s", settings.google_base_url)
    except Exception as e:
        logger.error("Failed to initialize Google OAuth: %s", e)
        logger.warning("Starting server without authentication")

# FastMCP instance (pass mask_error_details=True to avoid older settings attr lookup)
try:  # Prefer passing a truthy value so ToolManager won't access settings.mask_error_details
    mcp: FastMCP = FastMCP("crawler-mcp", mask_error_details=True, auth=auth_provider)  # type: ignore[call-arg]
except TypeError:  # Older FastMCP without this kwarg or auth param
    try:
        mcp = FastMCP("crawler-mcp", auth=auth_provider)
    except TypeError:
        mcp = FastMCP("crawler-mcp")

# Add required middlewares
mcp.add_middleware(ErrorHandlingMiddleware())
mcp.add_middleware(LoggingMiddleware())
mcp.add_middleware(TimingMiddleware())

# Register optimized tools for now
register_github_pr_tools(mcp)
register_rag_tools(mcp)
register_crawling_tools(mcp)

logger.info("Registered FastMCP tools (optimized)")


@mcp.tool
async def health_check(ctx: Context, detailed: bool = False) -> dict[str, Any]:
    """Perform a health check of all services."""
    from datetime import datetime
    from importlib import metadata as _metadata

    check_type = "detailed" if detailed else "lightweight"
    await ctx.info(f"Performing {check_type} health check of all services")

    try:
        health_results: dict[str, Any] = {
            "server": {
                "status": "healthy",
                "version": _metadata.version("crawler-mcp"),
                "timestamp": datetime.now(UTC).isoformat(),
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
                embedding_healthy = await embedding_service._health_check()
                info: dict[str, Any] = {
                    "status": "healthy" if embedding_healthy else "unhealthy",
                    "url": settings.tei_url,
                    "model": settings.tei_model,
                }
                if detailed:
                    info["model_info"] = await embedding_service.get_model_info(
                        max_size=2000
                    )
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


@mcp.tool
async def get_server_info(ctx: Context) -> dict[str, Any]:
    """Return effective configuration values to verify environment loading."""
    try:
        pkg_root = Path(__file__).parent
        info: dict[str, Any] = {
            "env_files_checked": {
                "package_env": str(pkg_root / ".env"),
                "project_root_env": str(pkg_root.parent / ".env"),
                "legacy_optimized_env": str(
                    pkg_root / "crawlers" / "optimized" / ".env"
                ),
            },
            "settings": {
                "tei_url": settings.tei_url,
                "tei_model": settings.tei_model,
                "qdrant_url": settings.qdrant_url,
                "qdrant_collection": settings.qdrant_collection,
                "qdrant_vector_size": settings.qdrant_vector_size,
                "max_concurrent_crawls": settings.max_concurrent_crawls,
                "log_level": settings.log_level,
            },
            "oauth": {
                "enabled": settings.oauth_enabled,
                "provider": settings.oauth_provider,
                "base_url": settings.google_base_url
                if settings.oauth_enabled
                else None,
                "scopes": settings.google_scopes_list
                if settings.oauth_enabled
                else None,
            },
            "optimized_config": OptimizedConfig.from_env().to_dict(),
            "env_present": {
                k: True
                for k in [
                    "TEI_URL",
                    "TEI_MODEL",
                    "QDRANT_URL",
                    "QDRANT_COLLECTION",
                    "QDRANT_API_KEY",
                    "MAX_CONCURRENT_CRAWLS",
                ]
                if os.environ.get(k) is not None
            },
            "optimized_env_present": sorted(
                k for k in os.environ if k.startswith("OPTIMIZED_CRAWLER_")
            ),
        }
        return info
    except Exception as e:
        raise ToolError(f"Failed to get server info: {e!s}") from e


@mcp.tool
async def get_user_info(ctx: Context) -> dict[str, Any]:
    """Returns information about the authenticated user (OAuth must be enabled)."""
    if not settings.oauth_enabled:
        raise ToolError("OAuth is not enabled on this server")

    try:
        from fastmcp.server.dependencies import get_access_token

        token = get_access_token()
        if not token:
            raise ToolError("No authentication token found")

        # Extract user info from token claims
        user_info = {
            "authenticated": True,
            "provider": settings.oauth_provider,
        }

        if settings.oauth_provider == "google":
            user_info.update(
                {
                    "google_id": token.claims.get("sub"),
                    "email": token.claims.get("email"),
                    "name": token.claims.get("name"),
                    "picture": token.claims.get("picture"),
                    "locale": token.claims.get("locale"),
                }
            )

        return user_info
    except ImportError:
        raise ToolError("Authentication dependencies not available") from None
    except Exception as e:
        raise ToolError(f"Failed to get user info: {e!s}") from e


def _sigterm_handler(signum: int, _frame: Any) -> None:  # pragma: no cover
    logger.info("Received signal %s, shutting down...", signum)
    sys.exit(0)


signal.signal(signal.SIGTERM, _sigterm_handler)


async def startup_checks() -> None:
    # Placeholders for any async startup checks in the future
    return None


def main() -> None:
    try:
        console = Console()
        console.print("")
        console.print("[bold blue]🕷️  Crawler-MCP Server v0.1.0[/bold blue]")
        console.print("[blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/blue]")
        console.print("[dim]RAG-Enabled Web Crawling Server[/dim]")
        console.print("")

        asyncio.run(startup_checks())

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
        sys.exit(0)
    except Exception as e:  # pragma: no cover
        logger.exception("Server failed to start: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
