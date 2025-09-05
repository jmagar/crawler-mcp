"""
Batch utility functions for optimized text processing.

This module provides reusable text batching algorithms extracted from the
optimized crawler's strategy.py to avoid duplication between the crawler and
webhook server.
"""

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


# Default configuration constants
DEFAULT_TARGET_CHARS = 64000  # Target character count per batch
DEFAULT_MAX_ITEMS = 128  # Maximum items per batch
DEFAULT_MIN_CHARS = 4000  # Minimum character count per batch


def pack_texts_into_batches(
    texts: list[str],
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_items: int = DEFAULT_MAX_ITEMS,
    min_chars: int = DEFAULT_MIN_CHARS,
    parallel_workers: int = 4,
) -> list[list[tuple[int, str]]]:
    """
    Pack texts into optimal batches using greedy algorithm from strategy.py.

    This is a direct extraction of the batching logic from lines 1468-1517 of strategy.py
    to enable code reuse between the crawler and webhook server.

    Args:
        texts: List of text strings to batch
        target_chars: Target character count per batch (default: 64000)
        max_items: Maximum items per batch (default: 128)
        min_chars: Minimum character count per batch (default: 4000)
        parallel_workers: Number of parallel workers to optimize for

    Returns:
        List of batches, where each batch is a list of (index, text) tuples

    Algorithm:
        1. Sorts texts by length (descending) for greedy packing
        2. Greedily packs into batches until reaching limits
        3. Splits large batches to ensure parallelism
    """
    if not texts:
        return []

    # Prepare (index, text) pairs and sort by length (desc) for greedy packing
    pairs = [(i, text) for i, text in enumerate(texts)]
    pairs.sort(key=lambda it: len(it[1]), reverse=True)

    # Greedy packing into batches
    batches: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    cur_chars = 0

    for pi, tx in pairs:
        tlen = len(tx)
        # If adding this would exceed limits, flush current batch
        if cur and (cur_chars + tlen > target_chars or len(cur) >= max_items):
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append((pi, tx))
        cur_chars += tlen

    if cur:
        batches.append(cur)

    # Ensure at least `parallel_workers` batches when possible to keep workers busy
    # by splitting the largest batches until we reach desired count.
    while len(batches) < parallel_workers and any(len(b) > 1 for b in batches):
        # find largest by chars
        li = max(range(len(batches)), key=lambda i: sum(len(t) for _, t in batches[i]))
        big = batches.pop(li)
        mid = len(big) // 2
        batches.append(big[:mid])
        batches.append(big[mid:])

    # Fallback if no batches created
    if not batches and texts:
        batches = [[(i, t)] for i, t in pairs]

    return batches


def pack_items_into_batches(
    items: list[T],
    text_extractor: "Callable[[T], str]",
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_items: int = DEFAULT_MAX_ITEMS,
    min_chars: int = DEFAULT_MIN_CHARS,
    parallel_workers: int = 4,
) -> list[list[tuple[int, T]]]:
    """
    Pack arbitrary items into optimal batches based on their text content.

    This is a generic version of pack_texts_into_batches that works with any item type
    by extracting text content using a provided function.

    Args:
        items: List of items to batch
        text_extractor: Function to extract text content from items (item -> str)
        target_chars: Target character count per batch
        max_items: Maximum items per batch
        min_chars: Minimum character count per batch
        parallel_workers: Number of parallel workers to optimize for
    Returns:
        List of batches, where each batch is a list of (index, item) tuples
    """
    if not items:
        return []

    # Extract texts for batching
    texts = [text_extractor(item) for item in items]

    # Use the core batching algorithm
    text_batches = pack_texts_into_batches(
        texts, target_chars, max_items, min_chars, parallel_workers
    )

    # Convert back to item batches
    item_batches = []
    for text_batch in text_batches:
        item_batch = [(idx, items[idx]) for idx, _ in text_batch]
        item_batches.append(item_batch)

    return item_batches


def calculate_batch_stats(batches: list[list[tuple[int, Any]]]) -> dict[str, Any]:
    """
    Calculate statistics for a set of batches.

    Args:
        batches: List of batches from pack_*_into_batches functions

    Returns:
        Dictionary with batch statistics
    """
    if not batches:
        return {
            "total_batches": 0,
            "total_items": 0,
            "avg_batch_size": 0,
            "avg_chars_per_batch": 0,
            "min_batch_size": 0,
            "max_batch_size": 0,
        }

    batch_sizes = [len(batch) for batch in batches]
    batch_chars = [sum(len(str(item)) for _, item in batch) for batch in batches]
    total_items = sum(batch_sizes)

    return {
        "total_batches": len(batches),
        "total_items": total_items,
        "avg_batch_size": total_items / len(batches) if batches else 0,
        "avg_chars_per_batch": sum(batch_chars) / len(batches) if batches else 0,
        "min_batch_size": min(batch_sizes) if batch_sizes else 0,
        "max_batch_size": max(batch_sizes) if batch_sizes else 0,
        "min_chars_per_batch": min(batch_chars) if batch_chars else 0,
        "max_chars_per_batch": max(batch_chars) if batch_chars else 0,
    }
