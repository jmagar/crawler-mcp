"""
Unified RAG module providing backward compatibility and clean API access.

This module maintains the original API while enabling access to the new modular structure.
Import patterns should work exactly as before for existing code.
"""

from typing import Any

# Main service class - primary API entry point
# Core processing components for advanced usage
from .chunking import (
    AdaptiveChunker,
    ChunkingStrategy,
    FixedSizeChunker,
    SemanticChunker,
    TokenBasedChunker,
    TokenCounter,
    find_line_boundary,
    find_paragraph_boundary,
    find_sentence_boundary,
    find_word_boundary,
)
from .deduplication import (
    ContentHasher,
    SimilarityDetector,
    VectorDeduplicationManager,
)
from .embedding import (
    EmbeddingCache,
    EmbeddingPipeline,
    EmbeddingWorker,
)
from .processing import (
    ProcessingPipeline,
    ProgressTracker,
    WorkflowManager,
)
from .service import (
    QueryCache,
    RagService,
    ServiceMetrics,
)

# Backward compatibility exports
# These maintain the original API for existing imports
__all__ = [
    "AdaptiveChunker",
    "ChunkingStrategy",
    "ContentHasher",
    "EmbeddingCache",
    "EmbeddingPipeline",
    "EmbeddingWorker",
    "FixedSizeChunker",
    "ProcessingPipeline",
    "ProgressTracker",
    "QueryCache",
    "RagService",
    "SemanticChunker",
    "ServiceMetrics",
    "SimilarityDetector",
    "TokenBasedChunker",
    "TokenCounter",
    "VectorDeduplicationManager",
    "WorkflowManager",
    "find_line_boundary",
    "find_paragraph_boundary",
    "find_sentence_boundary",
    "find_word_boundary",
]


def create_rag_service() -> RagService:
    """
    Factory function to create a RagService instance.

    This provides a clean entry point for creating the service
    and can be extended with configuration options in the future.

    Returns:
        Configured RagService instance
    """
    return RagService()


def get_available_chunking_strategies() -> list[str]:
    """
    Get list of available chunking strategy names.

    Returns:
        List of chunking strategy class names
    """
    return [
        "FixedSizeChunker",
        "TokenBasedChunker",
        "SemanticChunker",
        "AdaptiveChunker",
    ]


def create_chunking_strategy(strategy_name: str, **kwargs: Any) -> ChunkingStrategy:
    """
    Factory function to create chunking strategies by name.

    Args:
        strategy_name: Name of the chunking strategy
        **kwargs: Additional configuration parameters

    Returns:
        Configured chunking strategy instance

    Raises:
        ValueError: If strategy_name is not recognized
    """
    strategies = {
        "FixedSizeChunker": FixedSizeChunker,
        "TokenBasedChunker": TokenBasedChunker,
        "SemanticChunker": SemanticChunker,
        "AdaptiveChunker": AdaptiveChunker,
    }

    if strategy_name not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")

    return strategies[strategy_name](**kwargs)  # type: ignore[abstract]


def get_module_info() -> dict[str, str]:
    """
    Get information about the modular RAG structure.

    Returns:
        Dictionary with module descriptions
    """
    return {
        "chunking": "Text chunking strategies and boundary detection",
        "deduplication": "Content deduplication and hash management",
        "embedding": "Parallel embedding generation pipeline",
        "processing": "Main processing coordination and workflow management",
        "service": "Service orchestration, query processing, and caching",
    }


# Legacy compatibility - maintain exact import paths
# This ensures existing code like `from crawler_mcp.core.rag import RagService` still works
