from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.memory import MemoryManager
from ..core.logging import get_class_logger, get_logger
from ..models.crawl import CrawlResult

"""Base crawling strategy with common functionality."""

logger = get_logger(__name__)


class BaseCrawlStrategy(ABC):
    """
    Abstract base class for all crawling strategies.
    Defines the common interface and shared functionality.
    """

    def __init__(self) -> None:
        self.logger = get_class_logger(self)
        self.memory_manager: MemoryManager | None = None

    async def _initialize_managers(self) -> None:
        """Initializes the memory manager."""
        if not self.memory_manager:
            try:
                from ..core.memory import get_memory_manager
            except (ImportError, ModuleNotFoundError) as e:
                raise ImportError(
                    "Failed to import get_memory_manager from crawler_mcp.core.memory. "
                    "Please ensure the memory module is properly installed and the import path is correct."
                ) from e

            self.memory_manager = get_memory_manager()

    @abstractmethod
    async def execute(
        self,
        request: Any,  # Will be specific to each strategy
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> CrawlResult:
        """
        Execute the crawling strategy.

        Args:
            request: Strategy-specific request object
            progress_callback: Optional progress reporting callback

        Returns:
            CrawlResult with crawled data
        """
        pass

    @abstractmethod
    async def validate_request(self, request: Any) -> bool:
        """
        Validate that the request is valid for this strategy.

        Args:
            request: Strategy-specific request object

        Returns:
            True if request is valid
        """
        pass

    async def pre_execute_setup(self) -> None:
        """Setup tasks to run before crawling begins."""
        await self._initialize_managers()
        if self.memory_manager:
            await self.memory_manager.optimize_for_crawl()

    async def post_execute_cleanup(self) -> None:
        """Cleanup tasks to run after crawling completes."""
        if self.memory_manager:
            await self.memory_manager.check_memory_pressure(force_check=True)

    def _log_progress(self, message: str, level: str = "info") -> None:
        """Log progress messages with appropriate level."""
        if level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.info(message)
