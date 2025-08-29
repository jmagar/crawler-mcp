"""
Crawler MCP FastMCP Server - RAG-enabled web crawling with Crawl4AI and Qdrant.

This server provides comprehensive web crawling capabilities with automatic RAG indexing
using Crawl4AI 0.7.0, Qdrant vector database, and HF Text Embeddings Inference.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent fork warning with subprocesses

import warnings

# Suppress specific deprecation warnings from external dependencies FIRST
warnings.filterwarnings(
    "ignore", message=".*@validator.*is deprecated", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore",
    message="Support for class-based.*config.*is deprecated.*",
    category=DeprecationWarning,
)
# Suppress BeautifulSoup 'text' argument deprecation warnings only from bs4 module
warnings.filterwarnings(
    "ignore",
    message=r"The 'text' argument to find\(\)-type methods is deprecated\. Use 'string' instead\.",
    category=DeprecationWarning,
    module="bs4",
)

import asyncio  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from fastmcp import Context, FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402
from rich.traceback import install  # noqa: E402

# Configure logging before importing other modules
try:
    from .config import settings
    from .core import EmbeddingService, RagService, VectorService
    from .core.logging import get_logger
    from .tools.crawling import register_crawling_tools
    from .tools.rag import register_rag_tools
except ImportError:
    # Handle case when run as standalone script
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from crawler_mcp.config import settings
    from crawler_mcp.core import (
        EmbeddingService,
        RagService,
        VectorService,
    )
    from crawler_mcp.core.logging import get_logger
    from crawler_mcp.tools.crawling import register_crawling_tools
    from crawler_mcp.tools.rag import register_rag_tools


# Set up colorized logging configuration
def setup_logging() -> None:
    """Configure rich colorized logging for the application."""
    # Install rich tracebacks for better error display
    install(show_locals=settings.debug)

    # Create rich console
    console = Console(force_terminal=True, width=120)

    # Configure rich handler with colors matching dev.sh format
    rich_handler = RichHandler(
        console=console,
        show_path=settings.debug,
        show_time=True,
        rich_tracebacks=True,
        tracebacks_show_locals=settings.debug,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
    )

    # Create file handler if specified
    if settings.log_to_file and settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Add file handler with aligned format
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logging.getLogger().addHandler(file_handler)

    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("chromium").setLevel(logging.WARNING)


setup_logging()

logger = get_logger(__name__)


# Create FastMCP instance
mcp: FastMCP = FastMCP("crawler-mcp")

# Middleware will be handled by the HTTP transport automatically
# For now, we'll focus on the core functionality

# Register all tools
register_crawling_tools(mcp)
register_rag_tools(mcp)

logger.info("Registered all FastMCP tools")


@mcp.tool
async def health_check(ctx: Context, detailed: bool = False) -> dict[str, Any]:
    """
    Perform a health check of all services.

    Args:
        detailed: If True, includes comprehensive statistics. If False, returns lightweight status only.

    Returns:
        Dictionary with health status of all components
    """
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
                "chunk_size": 1000,  # From RAG service
                "vector_dimension": settings.qdrant_vector_size,
            },
        }
        services: dict[str, Any] = health_results["services"]

        # Check embedding service
        try:
            async with EmbeddingService() as embedding_service:
                embedding_healthy = await embedding_service.health_check()

                embedding_info = {
                    "status": "healthy" if embedding_healthy else "unhealthy",
                    "url": settings.tei_url,
                    "model": settings.tei_model,
                }

                if detailed:
                    model_info = await embedding_service.get_model_info(max_size=2000)
                    embedding_info["model_info"] = model_info

                services["embedding"] = embedding_info
        except Exception as e:
            services["embedding"] = {
                "status": "error",
                "error": str(e),
            }

        # Check vector service
        try:
            async with VectorService() as vector_service:
                vector_healthy = await vector_service.health_check()

                vector_info = {
                    "status": "healthy" if vector_healthy else "unhealthy",
                    "url": settings.qdrant_url,
                    "collection": settings.qdrant_collection,
                }

                if detailed:
                    collection_info = await vector_service.get_collection_info()
                    vector_info["collection_info"] = collection_info

                services["vector"] = vector_info
        except Exception as e:
            services["vector"] = {"status": "error", "error": str(e)}

        # Check RAG service (combines both)
        try:
            async with RagService() as rag_service:
                rag_health = await rag_service.health_check()

                rag_info = {
                    "status": "healthy" if all(rag_health.values()) else "unhealthy",
                    "component_health": rag_health,
                }

                if detailed:
                    rag_stats = await rag_service.get_stats()
                    rag_info["stats"] = rag_stats

                services["rag"] = rag_info
        except Exception as e:
            services["rag"] = {"status": "error", "error": str(e)}

        # Overall health status
        service_statuses = [
            service.get("status") == "healthy" for service in services.values()
        ]

        all_healthy = all(service_statuses)
        health_results["overall_status"] = "healthy" if all_healthy else "degraded"
        health_results["check_mode"] = check_type

        await ctx.info(
            f"{check_type.title()} health check completed - Overall status: {health_results['overall_status']}"
        )

        return health_results

    except Exception as e:
        error_msg = f"Health check failed: {e!s}"
        await ctx.info(error_msg)
        raise ToolError(error_msg) from e


@mcp.tool
async def get_server_info(ctx: Context) -> dict[str, Any]:
    """
    Get detailed information about the server configuration and capabilities.

    Returns:
        Dictionary with server information
    """
    await ctx.info("Retrieving server information")

    return {
        "server": {
            "name": "Crawler-MCP",
            "description": "RAG-enabled web crawling MCP server",
            "version": "0.1.0",
            "framework": "FastMCP 2.0+",
            "host": settings.server_host,
            "port": settings.server_port,
            "debug_mode": settings.debug,
            "production_mode": settings.production,
        },
        "capabilities": {
            "crawling": {
                "single_page_scraping": True,
                "website_crawling": True,
                "sitemap_support": True,
                "repository_cloning": True,
                "directory_processing": True,
                "max_concurrent_crawls": settings.max_concurrent_crawls,
                "supported_browsers": ["chromium"],
                "extraction_strategies": ["css", "llm", "cosine"],
            },
            "rag": {
                "semantic_search": True,
                "automatic_indexing": True,
                "chunk_processing": True,
                "reranking": True,
                "source_filtering": True,
                "embedding_model": settings.tei_model,
                "vector_dimension": settings.qdrant_vector_size,
                "distance_metric": settings.qdrant_distance,
            },
            "sources": {
                "source_management": True,
                "metadata_tracking": True,
                "filtering": True,
                "statistics": True,
                "supported_types": ["webpage", "repository", "directory"],
            },
        },
        "configuration": {
            "crawling": {
                "headless": settings.crawl_headless,
                "browser": settings.crawl_browser,
                "max_pages_default": settings.crawl_max_pages,
                "max_depth_default": settings.crawl_max_depth,
                "min_words": settings.crawl_min_words,
                "user_agent": settings.crawl_user_agent,
                "timeout": settings.crawler_timeout,
                "delay": settings.crawler_delay,
            },
            "embedding": {
                "service_url": settings.tei_url,
                "model": settings.tei_model,
                "batch_size": settings.tei_batch_size,
                "max_length": settings.embedding_max_length,
                "normalize": settings.embedding_normalize,
                "timeout": settings.tei_timeout,
            },
            "vector_database": {
                "service_url": settings.qdrant_url,
                "collection": settings.qdrant_collection,
                "vector_size": settings.qdrant_vector_size,
                "distance": settings.qdrant_distance,
                "timeout": settings.qdrant_timeout,
            },
        },
        "available_tools": [
            "scrape - Single page web scraping",
            "crawl - Multi-page website crawling",
            "crawl - Unified smart crawling (auto-detects websites, repositories, directories)",
            "rag_query - Semantic search queries",
            "list_sources - Source management",
            "health_check - System health monitoring",
            "get_server_info - Server information",
            "get_rag_stats - RAG system statistics",
            "delete_source - Source deletion",
        ],
    }


# Server lifecycle management will be handled by FastMCP automatically
async def startup_checks() -> None:
    """Initialize services and perform startup checks."""
    # Server info section
    logger.info("[bold cyan]▶ Configuration:[/bold cyan]")
    logger.info(
        "  • Mode: Debug %s | Production: %s", settings.debug, settings.production
    )
    logger.info("  • Endpoints:")
    logger.info("    - Qdrant: [blue]%s[/blue]", settings.qdrant_url)
    logger.info("    - TEI: [purple]%s[/purple]", settings.tei_url)
    logger.info("  • Models:")
    logger.info("    - Embedding: [magenta]%s[/magenta]", settings.tei_model)
    logger.info("    - Reranker: [yellow]%s[/yellow]", settings.reranker_model)
    logger.info("  • Timeout: [red]%ss[/red]", settings.crawler_timeout)
    logger.info("")

    # Health check section
    logger.info("[bold cyan]▶ Service Health:[/bold cyan]")

    try:
        # Check embedding service
        async with EmbeddingService() as embedding_service:
            embedding_healthy = await embedding_service.health_check()
            status = "✅" if embedding_healthy else "❌"
            logger.info("  %s Embedding service", status)

        # Check vector service
        async with VectorService() as vector_service:
            vector_healthy = await vector_service.health_check()
            collection_created = await vector_service.ensure_collection()
            v_status = "✅" if vector_healthy else "❌"
            c_status = "✅" if collection_created else "❌"
            logger.info("  %s Vector database", v_status)
            logger.info("  %s Collection ready", c_status)

        logger.info("")
        logger.info(
            "[bold green]✨ Crawler-MCP ready on %s:%s[/bold green]",
            settings.server_host,
            settings.server_port,
        )

    except ToolError:
        logger.exception("[red]❌ Critical startup error[/red]")
        logger.info("[dim]Server started but some services may be unavailable[/dim]")
    except (ConnectionError, TimeoutError) as e:
        logger.warning("  ❌ Service connection failed: %s", e)
        logger.info("[dim]Server started but some services may be unavailable[/dim]")
    except Exception:
        logger.exception("[red]❌ Unexpected startup error[/red]")
        logger.info("[dim]Server started but some services may be unavailable[/dim]")


# CLI entry point
def main() -> None:
    """Main entry point for the CLI."""
    try:
        # Rich startup banner
        console = Console()
        console.print("")
        console.print("[bold blue]🕷️  Crawler-MCP Server v0.1.0[/bold blue]")
        console.print("[blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/blue]")
        console.print("[dim]RAG-Enabled Web Crawling Server[/dim]")
        console.print("")

        # Run startup checks
        asyncio.run(startup_checks())

        # Import uvicorn for server
        import uvicorn

        # Start the server
        uvicorn.run(
            mcp.http_app(),
            host=settings.server_host,
            port=settings.server_port,
            log_level="info" if settings.debug else "warning",
        )

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Server failed to start: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
