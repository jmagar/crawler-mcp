"""
Base classes and mixins for common patterns in the Crawler MCP project.

This module provides reusable base classes to eliminate duplicate implementations
of common patterns like async context managers and singletons.
"""

import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

from .logging import get_class_logger

T = TypeVar("T", bound="AsyncContextManagerMixin")


class AsyncContextManagerMixin:
    """
    Base mixin for async context manager functionality.

    Provides standard __aenter__ and __aexit__ implementations
    that call abstract open() and close() methods.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = get_class_logger(self)
        self._is_open = False
        self._lock = asyncio.Lock()

    async def __aenter__(self: T) -> T:
        """Async context manager entry."""
        async with self._lock:
            if not self._is_open:
                await self.open()
                self._is_open = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        async with self._lock:
            if self._is_open:
                try:
                    await self.close()
                except Exception as e:
                    self.logger.warning(f"Error during close: {e}")
                finally:
                    self._is_open = False

    @abstractmethod
    async def open(self) -> None:
        """Open/initialize the resource. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close/cleanup the resource. Must be implemented by subclasses."""
        pass


class ServiceMixin:
    """
    Base mixin for service classes with common lifecycle methods.

    Provides standard implementations for health_check, cleanup, and initialization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = get_class_logger(self)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the service. Override in subclasses."""
        if self._initialized:
            return

        self.logger.debug(f"Initializing {self.__class__.__name__}")
        await self._initialize()
        self._initialized = True
        self.logger.debug(f"{self.__class__.__name__} initialized successfully")

    async def _initialize(self) -> None:
        """Service-specific initialization. Override in subclasses."""
        pass

    async def cleanup(self) -> None:
        """Cleanup service resources. Override in subclasses."""
        if not self._initialized:
            return

        self.logger.debug(f"Cleaning up {self.__class__.__name__}")
        try:
            await self._cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        finally:
            self._initialized = False
            self.logger.debug(f"{self.__class__.__name__} cleanup completed")

    async def _cleanup(self) -> None:
        """Service-specific cleanup. Override in subclasses."""
        pass

    async def health_check(self) -> bool:
        """Perform health check. Override in subclasses."""
        try:
            return await self._health_check()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def _health_check(self) -> bool:
        """Service-specific health check. Override in subclasses."""
        return self._initialized


class SingletonMixin:
    """
    Mixin to implement singleton pattern for services.

    Provides thread-safe singleton implementation with instance management.
    """

    _instances: ClassVar[dict[type, Any]] = {}
    _locks: ClassVar[dict[type, asyncio.Lock]] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            # Create lock if it doesn't exist
            if cls not in cls._locks:
                cls._locks[cls] = asyncio.Lock()

            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    @classmethod
    async def get_instance(cls) -> Any:
        """Get singleton instance asynchronously."""
        if cls not in cls._instances:
            if cls not in cls._locks:
                cls._locks[cls] = asyncio.Lock()

            async with cls._locks[cls]:
                if cls not in cls._instances:
                    instance = cls()
                    if hasattr(instance, "initialize"):
                        await instance.initialize()
                    cls._instances[cls] = instance

        return cls._instances[cls]

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        if cls in cls._instances:
            instance = cls._instances[cls]
            if hasattr(instance, "cleanup"):
                with contextlib.suppress(Exception):
                    await instance.cleanup()  # Ignore cleanup errors during reset
            del cls._instances[cls]

        if cls in cls._locks:
            del cls._locks[cls]


class AsyncServiceBase(AsyncContextManagerMixin, ServiceMixin, ABC):
    """
    Base class combining async context manager and service functionality.

    Perfect for services that need both lifecycle management and context manager support.
    """

    async def open(self) -> None:
        """Open service - delegates to initialize."""
        await self.initialize()

    async def close(self) -> None:
        """Close service - delegates to cleanup."""
        await self.cleanup()


class SingletonServiceBase(AsyncServiceBase, SingletonMixin, ABC):
    """
    Base class for singleton services with full lifecycle support.

    Combines singleton pattern, async context manager, and service lifecycle.
    """

    pass
