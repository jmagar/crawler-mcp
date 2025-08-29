"""
Async utilities and factories for creating and managing async primitives.

This module provides standardized factory functions and utilities for creating
asyncio primitives with consistent patterns and automatic resource management.
"""

import asyncio
import contextlib
import functools
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, TypeVar, cast

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class AsyncLockFactory:
    """Factory for creating and managing asyncio.Lock instances with automatic cleanup."""

    _locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _cleanup_tasks: ClassVar[dict[str, asyncio.Task[Any]]] = {}

    @classmethod
    def get_named_lock(cls, name: str) -> asyncio.Lock:
        """
        Get or create a named lock for shared resources.

        Args:
            name: Unique name for the lock

        Returns:
            asyncio.Lock instance
        """
        if name not in cls._locks:
            cls._locks[name] = asyncio.Lock()
            logger.debug(f"Created named lock: {name}")
        return cls._locks[name]

    @classmethod
    def get_class_lock(cls, class_type: type) -> asyncio.Lock:
        """
        Get or create a lock for a specific class (singleton pattern support).

        Args:
            class_type: The class type to create a lock for

        Returns:
            asyncio.Lock instance
        """
        name = f"{class_type.__module__}.{class_type.__name__}"
        return cls.get_named_lock(name)

    @classmethod
    def create_lock(cls) -> asyncio.Lock:
        """Create a new lock instance."""
        return asyncio.Lock()

    @classmethod
    def cleanup_named_locks(cls) -> None:
        """Clean up all named locks (for testing/shutdown)."""
        cls._locks.clear()
        for task in cls._cleanup_tasks.values():
            if not task.done():
                task.cancel()
        cls._cleanup_tasks.clear()


class AsyncQueueFactory:
    """Factory for creating various types of asyncio queues."""

    @staticmethod
    def create_queue(maxsize: int = 0) -> asyncio.Queue[Any]:
        """
        Create a standard FIFO queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)

        Returns:
            asyncio.Queue instance
        """
        return asyncio.Queue(maxsize=maxsize)

    @staticmethod
    def create_lifo_queue(maxsize: int = 0) -> asyncio.LifoQueue[Any]:
        """
        Create a LIFO (stack-like) queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)

        Returns:
            asyncio.LifoQueue instance
        """
        return asyncio.LifoQueue(maxsize=maxsize)

    @staticmethod
    def create_priority_queue(maxsize: int = 0) -> asyncio.PriorityQueue[Any]:
        """
        Create a priority queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)

        Returns:
            asyncio.PriorityQueue instance
        """
        return asyncio.PriorityQueue(maxsize=maxsize)


class AsyncSemaphoreFactory:
    """Factory for creating asyncio.Semaphore instances with common configurations."""

    @staticmethod
    def create_semaphore(value: int) -> asyncio.Semaphore:
        """
        Create a semaphore with specified value.

        Args:
            value: Initial semaphore value

        Returns:
            asyncio.Semaphore instance
        """
        return asyncio.Semaphore(value)

    @staticmethod
    def create_bounded_semaphore(value: int) -> asyncio.BoundedSemaphore:
        """
        Create a bounded semaphore (prevents over-release).

        Args:
            value: Initial semaphore value

        Returns:
            asyncio.BoundedSemaphore instance
        """
        return asyncio.BoundedSemaphore(value)

    @staticmethod
    def create_rate_limiter(requests_per_second: float) -> asyncio.Semaphore:
        """
        Create a semaphore configured for rate limiting.

        Args:
            requests_per_second: Maximum requests per second

        Returns:
            asyncio.Semaphore instance
        """
        # Allow burst of requests up to rate limit
        return asyncio.Semaphore(max(1, int(requests_per_second)))


class AsyncEventFactory:
    """Factory for creating asyncio.Event instances."""

    @staticmethod
    def create_event() -> asyncio.Event:
        """Create a new event."""
        return asyncio.Event()

    @staticmethod
    def create_condition() -> asyncio.Condition:
        """Create a new condition variable."""
        return asyncio.Condition()


class TaskManager:
    """Manager for creating and tracking background tasks."""

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[Any]] = []
        self._lock = asyncio.Lock()

    async def create_task(
        self, coro: Awaitable[T], name: str | None = None, track: bool = True
    ) -> asyncio.Task[T]:
        """
        Create a task with optional tracking.

        Args:
            coro: Coroutine to run
            name: Optional name for the task
            track: Whether to track the task for cleanup

        Returns:
            asyncio.Task instance
        """
        task: asyncio.Task[T] = asyncio.create_task(
            cast(Coroutine[Any, Any, T], coro), name=name
        )

        if track:
            async with self._lock:
                self._tasks.append(task)

            # Auto-cleanup completed tasks
            def cleanup_completed(t: asyncio.Task[Any]) -> None:
                if t.done():
                    cleanup_task = asyncio.create_task(self._remove_task(t))
                    # Store reference to prevent garbage collection warning
                    AsyncLockFactory._cleanup_tasks[f"cleanup_{id(t)}"] = cleanup_task

            task.add_done_callback(cleanup_completed)

        return task

    async def _remove_task(self, task: asyncio.Task[Any]) -> None:
        """Remove a task from tracking."""
        async with self._lock:
            if task in self._tasks:
                self._tasks.remove(task)

    async def cancel_all(self) -> None:
        """Cancel all tracked tasks."""
        async with self._lock:
            tasks = self._tasks.copy()

        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_for_completion(self, timeout: float | None = None) -> None:
        """Wait for all tracked tasks to complete."""
        async with self._lock:
            tasks = self._tasks.copy()

        if tasks:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
                )
            else:
                await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def active_count(self) -> int:
        """Get the number of active tracked tasks."""
        return len([t for t in self._tasks if not t.done()])


# Global task manager instance
_task_manager = TaskManager()


def create_background_task(
    coro: Awaitable[T], name: str | None = None, track: bool = True
) -> asyncio.Task[T]:
    """
    Create a background task with automatic tracking.

    Args:
        coro: Coroutine to run
        name: Optional name for the task
        track: Whether to track for cleanup

    Returns:
        asyncio.Task instance
    """
    # Create the task directly and then add tracking if needed
    task: asyncio.Task[T] = asyncio.create_task(
        cast(Coroutine[Any, Any, T], coro), name=name
    )

    if track:
        # Add tracking in a separate task to avoid blocking
        def track_task() -> None:
            async def add_tracking() -> None:
                async with _task_manager._lock:
                    _task_manager._tasks.append(task)

            tracking_task = asyncio.create_task(add_tracking())
            # Store reference to prevent garbage collection
            tracking_task.add_done_callback(lambda t: None)

        track_task()

    return task


async def run_with_semaphore(
    semaphore: asyncio.Semaphore,
    coro_func: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Run a coroutine function with semaphore protection.

    Args:
        semaphore: Semaphore to use for limiting concurrency
        coro_func: Async function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the coroutine function
    """
    async with semaphore:
        return await coro_func(*args, **kwargs)


async def run_parallel_limited(
    coro_funcs: list[Callable[..., Awaitable[T]]],
    max_concurrency: int,
    *args: Any,
    **kwargs: Any,
) -> list[T]:
    """
    Run multiple coroutines with limited concurrency.

    Args:
        coro_funcs: List of async functions to call
        max_concurrency: Maximum number of concurrent operations
        *args: Positional arguments for all functions
        **kwargs: Keyword arguments for all functions

    Returns:
        List of results from all functions
    """
    semaphore = AsyncSemaphoreFactory.create_semaphore(max_concurrency)
    tasks = [
        run_with_semaphore(semaphore, func, *args, **kwargs) for func in coro_funcs
    ]
    return await asyncio.gather(*tasks)


@contextlib.asynccontextmanager
async def timeout_context(timeout: float) -> AsyncGenerator[None, None]:
    """
    Async context manager for timeout operations.

    Args:
        timeout: Timeout in seconds

    Usage:
        async with timeout_context(5.0):
            result = await some_operation()
    """
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.Event().wait(), timeout=0)  # Start timeout

    try:
        yield
    except TimeoutError:
        logger.warning(f"Operation timed out after {timeout} seconds")
        raise


def retry_async(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    logger.warning(
                        f"Function {func.__name__} attempt {attempt + 1} failed: {e}, retrying in {current_delay}s"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            return None  # type: ignore

        return wrapper

    return decorator


@contextlib.asynccontextmanager
async def thread_pool_executor(
    max_workers: int | None = None,
) -> AsyncGenerator[ThreadPoolExecutor, None]:
    """
    Async context manager for ThreadPoolExecutor.

    Args:
        max_workers: Maximum number of worker threads

    Usage:
        async with thread_pool_executor(4) as executor:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, cpu_bound_func, arg)
    """
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


async def run_in_thread_pool(
    func: Callable[..., T], *args: Any, max_workers: int | None = None, **kwargs: Any
) -> T:
    """
    Run a synchronous function in a thread pool.

    Args:
        func: Function to run in thread pool
        *args: Positional arguments for the function
        max_workers: Maximum number of worker threads
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the function
    """
    target_func: Callable[..., T]
    if kwargs:
        # If we have keyword arguments, create a partial function
        partial_func = functools.partial(func, **kwargs)
        target_func = partial_func
        target_args = args
    else:
        target_func = func
        target_args = args

    async with thread_pool_executor(max_workers) as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, target_func, *target_args)


class AsyncResourceManager:
    """Generic resource manager for async resources with automatic cleanup."""

    def __init__(self) -> None:
        self._resources: list[Any] = []
        self._cleanup_funcs: list[Callable[[], Any] | None] = []
        self._lock = asyncio.Lock()

    async def add_resource(
        self, resource: Any, cleanup_func: Callable[[], Any] | None = None
    ) -> None:
        """
        Add a resource to be managed.

        Args:
            resource: Resource to manage
            cleanup_func: Optional cleanup function
        """
        async with self._lock:
            self._resources.append(resource)
            if cleanup_func:
                self._cleanup_funcs.append(cleanup_func)
            elif hasattr(resource, "close"):
                self._cleanup_funcs.append(resource.close)
            elif hasattr(resource, "aclose"):
                self._cleanup_funcs.append(resource.aclose)
            else:
                self._cleanup_funcs.append(None)

    async def cleanup_all(self) -> None:
        """Clean up all managed resources."""
        async with self._lock:
            for resource, cleanup_func in zip(
                self._resources, self._cleanup_funcs, strict=False
            ):
                if cleanup_func:
                    try:
                        if inspect.iscoroutinefunction(cleanup_func):
                            await cleanup_func()
                        else:
                            cleanup_func()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup resource {resource}: {e}")

            self._resources.clear()
            self._cleanup_funcs.clear()

    async def __aenter__(self) -> "AsyncResourceManager":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup_all()


# Convenience functions for common patterns
create_lock = AsyncLockFactory.create_lock
get_named_lock = AsyncLockFactory.get_named_lock
get_class_lock = AsyncLockFactory.get_class_lock
create_queue = AsyncQueueFactory.create_queue
create_semaphore = AsyncSemaphoreFactory.create_semaphore
create_event = AsyncEventFactory.create_event


async def shutdown_async_utils() -> None:
    """Clean up all async utilities (for application shutdown)."""
    await _task_manager.cancel_all()
    AsyncLockFactory.cleanup_named_locks()
    logger.info("Async utilities shutdown completed")
