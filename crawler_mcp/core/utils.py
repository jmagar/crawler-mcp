"""
Common utilities for the Crawler MCP project.

This module consolidates duplicate utility functions found throughout
the codebase to eliminate code duplication and provide consistent behavior.
"""

import contextlib
import os
import re
import sys
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from .logging import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def suppress_stdout() -> Any:
    """
    Context manager to suppress stdout output (redirect to devnull).

    Prevents stdout interference with MCP protocol communication.
    Consolidated from duplicate implementations in web.py and orchestrator.py.

    Usage:
        with suppress_stdout():
            print("This won't be visible")  # Suppressed
    """
    old_stdout = sys.stdout
    try:
        # Redirect stdout to devnull to prevent interference with MCP protocol
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = old_stdout


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent hashing and comparison.

    Consolidated from duplicate implementations in deduplication.py and service.py.

    Normalizations applied:
    - Protocol (http -> https)
    - Removes trailing slashes from path
    - Sorts query parameters alphabetically
    - Removes fragments
    - Lowercases domain

    Args:
        url: URL to normalize

    Returns:
        Normalized URL string

    Examples:
        >>> normalize_url("http://example.com/path/?b=2&a=1#fragment")
        "https://example.com/path?a=1&b=2"

        >>> normalize_url("https://EXAMPLE.COM/path//")
        "https://example.com/path"
    """
    try:
        parsed = urlparse(url)

        # Normalize protocol to https (only for http/https schemes)
        scheme = "https" if parsed.scheme in ("http", "https") else parsed.scheme

        # Remove trailing slash from path
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        # Sort query parameters for consistency
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            sorted_params = sorted(params.items())
            query = urlencode(sorted_params, doseq=True)
        else:
            query = ""

        # Reconstruct normalized URL (without fragment)
        normalized = urlunparse(
            (
                scheme,
                parsed.netloc.lower(),  # Lowercase domain
                path,
                parsed.params,
                query,
                "",  # Remove fragment
            )
        )

        return normalized

    except Exception as e:
        logger.warning(f"Failed to normalize URL '{url}': {e}")
        return url  # Return original URL if normalization fails


def normalize_whitespace(content: str) -> str:
    """
    Normalize whitespace in content for consistent processing.

    Moved from deduplication.py to eliminate single-use utility.

    Normalizations applied:
    - Strips leading and trailing whitespace
    - Replaces multiple consecutive whitespace characters with single space
    - Preserves intentional spacing in formatted content

    Args:
        content: Content string to normalize

    Returns:
        Content with normalized whitespace

    Examples:
        >>> normalize_whitespace("  Hello    world  \\n\\t  ")
        "Hello world"

        >>> normalize_whitespace("Multiple\\n\\nlines\\t\\twith    spaces")
        "Multiple lines with spaces"
    """
    if not content:
        return ""

    # Replace multiple whitespace characters with single space
    normalized = re.sub(r"\s+", " ", content.strip())
    return normalized


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Sanitize filename by removing or replacing invalid characters.

    Removes characters that are invalid in filenames across different
    operating systems and replaces them with the specified replacement.

    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with (default: "_")

    Returns:
        Sanitized filename safe for use across OS platforms

    Examples:
        >>> sanitize_filename("file<>name?.txt")
        "file__name_.txt"

        >>> sanitize_filename("valid-file_name.txt")
        "valid-file_name.txt"
    """
    if not filename:
        return "unnamed_file"

    # Characters invalid in filenames on various OS
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'

    # Replace invalid characters
    sanitized = re.sub(invalid_chars, replacement, filename)

    # Remove leading/trailing periods and spaces
    sanitized = sanitized.strip(". ")

    # Ensure we don't have an empty filename
    if not sanitized:
        sanitized = "unnamed_file"

    # Limit length to reasonable size (255 is typical filesystem limit)
    if len(sanitized) > 200:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[: 200 - len(ext)] + ext

    return sanitized


def clean_text_content(text: str) -> str:
    """
    Clean text content by removing extra whitespace and control characters.

    Combines whitespace normalization with control character removal
    for clean text processing.

    Args:
        text: Raw text content

    Returns:
        Cleaned text content

    Examples:
        >>> clean_text_content("\\x00Hello\\r\\n\\tworld\\x01\\n\\n")
        "Hello world"
    """
    if not text:
        return ""

    # Remove control characters (except common whitespace)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize whitespace
    cleaned = normalize_whitespace(cleaned)

    return cleaned


def truncate_with_ellipsis(text: str, max_length: int, ellipsis: str = "...") -> str:
    """
    Truncate text to specified length with ellipsis indicator.

    Ensures text doesn't exceed max_length while providing clear indication
    when content has been truncated.

    Args:
        text: Text to potentially truncate
        max_length: Maximum allowed length including ellipsis
        ellipsis: String to append when truncating (default: "...")

    Returns:
        Original text or truncated text with ellipsis

    Examples:
        >>> truncate_with_ellipsis("This is a very long text", 15)
        "This is a ve..."

        >>> truncate_with_ellipsis("Short text", 15)
        "Short text"
    """
    if not text or max_length <= 0:
        return ""

    if len(text) <= max_length:
        return text

    if len(ellipsis) >= max_length:
        return ellipsis[:max_length]

    truncate_at = max_length - len(ellipsis)
    return text[:truncate_at] + ellipsis
