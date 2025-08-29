"""
Centralized exception handling for the Crawler MCP project.

This module provides base exception classes and decorators to eliminate
duplicate exception handling patterns throughout the application.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from .logging import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CrawlerMCPError(Exception):
    """Base exception for all Crawler MCP related errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


class CrawlError(CrawlerMCPError):
    """Exception raised during crawling operations."""

    pass


class ConfigurationError(CrawlerMCPError):
    """Exception raised for configuration-related errors."""

    pass


class ServiceError(CrawlerMCPError):
    """Exception raised for service-related errors."""

    pass


class ValidationError(CrawlerMCPError):
    """Exception raised for validation errors."""

    pass


def handle_exceptions(
    *,
    logger_instance: logging.Logger | None = None,
    default_return: Any = None,
    re_raise: bool = False,
    log_level: int = logging.ERROR,
    message_template: str = "Error in {function_name}: {error}",
) -> Callable[[Callable[P, T]], Callable[P, T | Any]]:
    """
    Decorator to centralize exception handling patterns.

    Args:
        logger_instance: Optional specific logger to use
        default_return: Value to return if exception occurs and re_raise is False
        re_raise: Whether to re-raise the exception after logging
        log_level: Logging level for the error message
        message_template: Template for the error message

    Returns:
        Decorated function with centralized exception handling
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | Any]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | Any:
            current_logger = logger_instance or logger
            function_name = f"{func.__module__}.{func.__qualname__}"

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = message_template.format(
                    function_name=function_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                current_logger.log(log_level, error_message)

                if re_raise:
                    raise
                return default_return

        return wrapper

    return decorator


def handle_async_exceptions(
    *,
    logger_instance: logging.Logger | None = None,
    default_return: Any = None,
    re_raise: bool = False,
    log_level: int = logging.ERROR,
    message_template: str = "Error in {function_name}: {error}",
) -> Callable[[Callable[P, T]], Callable[P, T | Any]]:
    """
    Async version of handle_exceptions decorator.

    Args:
        logger_instance: Optional specific logger to use
        default_return: Value to return if exception occurs and re_raise is False
        re_raise: Whether to re-raise the exception after logging
        log_level: Logging level for the error message
        message_template: Template for the error message

    Returns:
        Decorated async function with centralized exception handling
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | Any]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | Any:
            current_logger = logger_instance or logger
            function_name = f"{func.__module__}.{func.__qualname__}"

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_message = message_template.format(
                    function_name=function_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                current_logger.log(log_level, error_message)

                if re_raise:
                    raise
                return default_return

        return wrapper

    return decorator


def log_and_suppress_exceptions(
    logger_instance: logging.Logger | None = None,
    message: str = "Exception suppressed",
    log_level: int = logging.WARNING,
) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """
    Decorator to log exceptions and suppress them (returns None).
    Useful for cleanup operations where exceptions should not propagate.

    Args:
        logger_instance: Optional specific logger to use
        message: Message to log with the exception
        log_level: Logging level for the message

    Returns:
        Decorated function that suppresses exceptions
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            current_logger = logger_instance or logger

            try:
                return func(*args, **kwargs)
            except Exception as e:
                current_logger.log(log_level, f"{message}: {e}")
                return None

        return wrapper

    return decorator
