"""
Centralized logging factory for the Crawler MCP project.

This module provides a consistent way to create and configure loggers throughout
the application, eliminating duplicate logger creation patterns.
"""

import logging
from typing import Any


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """
    Create or get a logger with the specified name.

    Args:
        name: Logger name, typically __name__ from the calling module
        level: Optional logging level override

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


def get_class_logger(class_instance: Any) -> logging.Logger:
    """
    Create a logger for a class instance with a descriptive name.

    Args:
        class_instance: Instance of the class that needs a logger

    Returns:
        Logger with name format: module.ClassName
    """
    module_name = class_instance.__class__.__module__
    class_name = class_instance.__class__.__name__
    logger_name = f"{module_name}.{class_name}"

    return get_logger(logger_name)


def configure_logging(
    level: int = logging.INFO, format_string: str | None = None
) -> None:
    """
    Configure global logging settings.

    Args:
        level: Default logging level
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level, format=format_string, handlers=[logging.StreamHandler()]
    )
