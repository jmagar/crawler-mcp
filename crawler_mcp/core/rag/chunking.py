"""
Text chunking strategies and token counting for RAG operations.

This module provides various chunking strategies for breaking down text content
into optimal chunks for embedding generation and vector search.
"""

import logging
import re
import threading
from abc import ABC, abstractmethod
from typing import Any

from crawler_mcp.settings import get_settings

settings = get_settings()


def safe_strip(content: Any) -> str:
    """Safely strip content, handling lists and other types."""
    if content is None:
        return ""
    if isinstance(content, list):
        return "\n".join(str(item) for item in content).strip()
    return str(content).strip()


logger = logging.getLogger(__name__)

# Approximate word-to-token ratio for different tokenizers
WORD_TO_TOKEN_RATIO = 1.3  # General estimate for English text
QWEN3_WORD_TO_TOKEN_RATIO = 1.4  # Qwen3 tokenizer ratio


def find_paragraph_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find paragraph break boundary."""
    paragraph_breaks = [
        i for i, char in enumerate(search_text) if search_text[i : i + 2] == "\n\n"
    ]
    suitable_breaks = [
        b for b in paragraph_breaks if ideal_end - 100 <= b <= ideal_end + 100
    ]
    return max(suitable_breaks) if suitable_breaks else None


def find_sentence_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find sentence ending boundary."""
    sentence_patterns = [". ", "! ", "? ", ".\n", "!\n", "?\n", ".", "!", "?"]
    sentence_breaks = []
    for pattern in sentence_patterns:
        sentence_breaks.extend(
            [
                i + len(pattern)
                for i in range(len(search_text) - len(pattern) + 1)
                if search_text[i : i + len(pattern)] == pattern
            ]
        )
    suitable_breaks = [
        b for b in sentence_breaks if ideal_end - 50 <= b <= ideal_end + 50
    ]
    return max(suitable_breaks) if suitable_breaks else None


def find_line_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find line break boundary."""
    line_breaks = [i + 1 for i, char in enumerate(search_text) if char == "\n"]
    suitable_breaks = [b for b in line_breaks if ideal_end - 30 <= b <= ideal_end + 30]
    return max(suitable_breaks) if suitable_breaks else None


def find_word_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find word boundary."""
    word_breaks = [i + 1 for i, char in enumerate(search_text) if char == " "]
    suitable_breaks = [b for b in word_breaks if ideal_end - 20 <= b <= ideal_end + 20]
    return max(suitable_breaks) if suitable_breaks else None


class TokenCounter:
    """Accurate token counting with multiple tokenizer support."""

    # Class-level tokenizer caching to prevent repeated downloads
    _tokenizer: Any | None = None
    _tokenizer_type: str = "word-estimate"
    _tokenizer_lock: threading.Lock = threading.Lock()
    _tokenizer_initialized: bool = False

    def __init__(self) -> None:
        # Use configurable word_to_token_ratio from settings with fallback
        try:
            configured_ratio = getattr(
                settings, "word_to_token_ratio", QWEN3_WORD_TO_TOKEN_RATIO
            )
            if isinstance(configured_ratio, int | float) and configured_ratio > 0:
                self.word_to_token_ratio = configured_ratio
            else:
                logger.warning(
                    f"Invalid word_to_token_ratio {configured_ratio}, using default {QWEN3_WORD_TO_TOKEN_RATIO}"
                )
                self.word_to_token_ratio = QWEN3_WORD_TO_TOKEN_RATIO
        except Exception as e:
            logger.warning(
                f"Error loading word_to_token_ratio from settings: {e}, using default {QWEN3_WORD_TO_TOKEN_RATIO}"
            )
            self.word_to_token_ratio = QWEN3_WORD_TO_TOKEN_RATIO

        # Initialize tokenizer once at class level
        self._ensure_tokenizer_initialized()

        # Set instance variables from class-level cache
        self.tokenizer = TokenCounter._tokenizer
        self.tokenizer_type = TokenCounter._tokenizer_type

    @classmethod
    def _ensure_tokenizer_initialized(cls) -> None:
        """Initialize tokenizer once per class, thread-safe."""
        if cls._tokenizer_initialized:
            return

        with cls._tokenizer_lock:
            # Double-check pattern - another thread might have initialized while we waited
            if cls._tokenizer_initialized:
                return

            try:
                from transformers import AutoTokenizer

                # Use the same model as your embedding service
                model_name = getattr(settings, "tei_model", "Qwen/Qwen3-Embedding-0.6B")
                model_revision = getattr(settings, "tei_model_revision", None)

                tokenizer_kwargs = {
                    "local_files_only": True,
                    "trust_remote_code": False,
                }
                if model_revision:
                    tokenizer_kwargs["revision"] = model_revision

                cls._tokenizer = AutoTokenizer.from_pretrained(
                    model_name, **tokenizer_kwargs
                )
                cls._tokenizer_type = "qwen3"
                logger.info(
                    f"Initialized shared Qwen3 tokenizer from {model_name} (revision: {model_revision or 'main'})"
                )
            except ImportError:
                logger.info(
                    "transformers not available (install with [ml] extra); using word-based estimation"
                )
                cls._tokenizer = None
                cls._tokenizer_type = "word-estimate"
            except Exception as e:
                logger.warning(
                    f"Failed to load Qwen3 tokenizer: {e}; using word-based estimation"
                )
                cls._tokenizer = None
                cls._tokenizer_type = "word-estimate"
            finally:
                cls._tokenizer_initialized = True

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Qwen3 tokenizer or word-based estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Accurate token count
        """
        if not safe_strip(text):
            return 0

        if self.tokenizer and self.tokenizer_type == "qwen3":
            try:
                # Use Qwen3 tokenizer for accurate count
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(
                    f"Qwen3 tokenizer failed: {e}, falling back to estimation"
                )

        # Fallback to word-based estimation using Qwen3's ratio
        return self.estimate_tokens_from_words(text)

    def estimate_tokens_from_words(self, text: str) -> int:
        """Estimate token count from word count."""
        word_count = len(text.split())
        return int(word_count * self.word_to_token_ratio)

    def calculate_chunk_token_distribution(self, chunks: list[str]) -> dict[str, float]:
        """Calculate token distribution statistics for chunks."""
        if not chunks:
            return {}

        token_counts = [self.count_tokens(chunk) for chunk in chunks]
        return {
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "total_tokens": sum(token_counts),
        }


class ChunkingStrategy(ABC):
    """Base class for text chunking strategies."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        # Validate chunk_size
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        # Validate overlap
        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError("overlap must be a non-negative integer")

        # Validate relationship between chunk_size and overlap
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.token_counter = TokenCounter()

    @abstractmethod
    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Chunk text using the specific strategy.

        Args:
            text: Text to chunk
            metadata: Optional metadata to include in chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        pass

    def get_chunk_metric(self) -> str:
        """Get the metric used for chunk size (characters or tokens)."""
        return "characters"  # Default to characters

    def measure_chunk(self, chunk: str) -> int:
        """Measure chunk size using the appropriate metric."""
        return len(chunk)  # Default to character count

    def get_min_chunk_size(self) -> int:
        """Get minimum allowed chunk size in the appropriate metric."""
        return 50  # Default minimum characters

    def validate_chunk_size(self, chunk: str) -> bool:
        """
        Validate if chunk size is within acceptable limits.

        Uses strategy-appropriate validation based on the chunk metric.

        Args:
            chunk: Text chunk to validate

        Returns:
            True if chunk size is valid, False otherwise
        """
        actual_size = self.measure_chunk(chunk)
        min_size = self.get_min_chunk_size()
        max_size = self.chunk_size * 2  # Maximum is double the target

        return min_size < actual_size <= max_size

    def assess_chunk_quality(
        self, chunk: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """Assess chunk quality across multiple dimensions."""
        return {
            "completeness": self._assess_completeness(chunk),
            "coherence": self._assess_coherence(chunk),
            "information_density": self._assess_information_density(chunk),
            "context_preservation": self._assess_context_preservation(
                chunk, metadata or {}
            ),
        }

    def _assess_completeness(self, chunk: str) -> float:
        """Assess how complete the chunk is (sentences end properly, etc.)."""
        if not safe_strip(chunk):
            return 0.0

        # Check if chunk ends with sentence punctuation
        stripped_chunk = safe_strip(chunk)
        if stripped_chunk:
            last_char = stripped_chunk[-1]
            if last_char in ".!?":
                return 1.0
            elif last_char in ",;:":
                return 0.7
        else:
            return 0.5

    def _assess_coherence(self, chunk: str) -> float:
        """Assess how coherent the chunk is."""
        # Simple heuristic: longer chunks with proper sentence structure are more coherent
        sentences = chunk.split(".")
        if len(sentences) < 2:
            return 0.5

        avg_sentence_length = sum(len(safe_strip(s).split()) for s in sentences) / len(
            sentences
        )
        if 5 <= avg_sentence_length <= 25:  # Reasonable sentence length
            return 1.0
        else:
            return 0.7

    def _assess_information_density(self, chunk: str) -> float:
        """Assess information density of the chunk."""
        words = chunk.split()
        if not words:
            return 0.0

        # Heuristic: ratio of unique words to total words
        unique_words = {word.lower() for word in words}
        density = len(unique_words) / len(words)
        return min(1.0, density * 2)  # Scale to 0-1 range

    def _assess_context_preservation(
        self, chunk: str, metadata: dict[str, Any]
    ) -> float:
        """Assess how well context is preserved in the chunk."""
        # If we have overlap, context is better preserved
        if self.overlap > 0:
            return 1.0
        else:
            return 0.8


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking with overlap and smart boundary detection."""

    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Optimized character-based chunking with semantic boundary detection.
        """
        chunks = []
        text_length = len(text)
        start = 0
        chunk_index = 0

        while start < text_length:
            end = start + self.chunk_size

            # Find a good break point using boundary detection helpers
            if end < text_length:
                # Get search window for boundary detection
                search_start = max(start, end - 100)
                search_end = min(text_length, end + 100)
                search_text = text[search_start:search_end]
                relative_ideal = end - search_start

                # Try different boundary types in order of preference
                boundary = find_paragraph_boundary(search_text, relative_ideal)
                if boundary is not None:
                    end = search_start + boundary
                else:
                    boundary = find_sentence_boundary(search_text, relative_ideal)
                    if boundary is not None:
                        end = search_start + boundary
                    else:
                        boundary = find_line_boundary(search_text, relative_ideal)
                        if boundary is not None:
                            end = search_start + boundary
                        else:
                            boundary = find_word_boundary(search_text, relative_ideal)
                            if boundary is not None:
                                end = search_start + boundary

            chunk_text = safe_strip(text[start:end])

            if chunk_text:
                chunk = {
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text),
                    "token_count_estimate": self.token_counter.count_tokens(chunk_text),
                    **(metadata or {}),
                }
                chunks.append(chunk)
                chunk_index += 1

            # Move start from actual end to preserve overlap
            start = max(0, end - self.overlap)

        return chunks


class TokenBasedChunker(ChunkingStrategy):
    """Token-based chunking using actual tokenizer for optimal semantic boundaries."""

    def get_chunk_metric(self) -> str:
        """Get the metric used for chunk size."""
        return "tokens"

    def measure_chunk(self, chunk: str) -> int:
        """Measure chunk size in tokens."""
        return self.token_counter.count_tokens(chunk)

    def get_min_chunk_size(self) -> int:
        """Get minimum allowed chunk size in tokens."""
        return 10  # Minimum tokens

    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Token-based chunking using tokenizer for optimal semantic boundaries.
        """
        chunks = []

        if self.token_counter.tokenizer:
            # Use actual tokenizer (Qwen3 or fallback)
            if self.token_counter.tokenizer_type == "qwen3":
                # Qwen3 tokenizer from transformers
                tokens = self.token_counter.tokenizer.encode(
                    text, add_special_tokens=False
                )
            else:
                # Other tokenizers (e.g., tiktoken if available)
                tokens = self.token_counter.tokenizer.encode(text)

            total_tokens = len(tokens)
            start_token = 0
            chunk_index = 0

            while start_token < total_tokens:
                end_token = min(start_token + self.chunk_size, total_tokens)

                # Extract token chunk
                chunk_tokens = tokens[start_token:end_token]

                # Decode based on tokenizer type
                if self.token_counter.tokenizer_type == "qwen3":
                    chunk_text = self.token_counter.tokenizer.decode(
                        chunk_tokens, skip_special_tokens=True
                    )
                else:
                    chunk_text = self.token_counter.tokenizer.decode(chunk_tokens)

                if safe_strip(chunk_text):
                    chunk = {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "start_pos": start_token,
                        "end_pos": end_token,
                        "start_token": start_token,
                        "end_token": end_token,
                        "token_count": len(chunk_tokens),
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        **(metadata or {}),
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                # Advance window maintaining stable overlap
                # Ensure we advance by at least 1 token to avoid infinite loop
                next_start = end_token - self.overlap
                start_token = max(start_token + 1, min(next_start, total_tokens))
        else:
            # Fallback to approximate token-based chunking using word estimation
            # Build word position map to track actual character offsets
            word_pattern = re.compile(r"\S+")
            word_positions = []
            for match in word_pattern.finditer(text):
                word_positions.append(
                    {"word": match.group(), "start": match.start(), "end": match.end()}
                )

            total_words = len(word_positions)
            # Use configurable word-to-token ratio for accuracy
            approx_tokens_per_word = self.token_counter.word_to_token_ratio
            # Ensure at least one word per chunk and valid overlap relation
            chunk_size_words = max(1, round(self.chunk_size / approx_tokens_per_word))
            overlap_words = max(
                0,
                min(
                    round(self.overlap / approx_tokens_per_word),
                    chunk_size_words - 1,
                ),
            )

            start_word = 0
            chunk_index = 0

            while start_word < total_words:
                end_word = min(start_word + chunk_size_words, total_words)

                if start_word >= total_words:
                    break

                # Extract chunk using actual positions from original text
                text_start_pos: int = word_positions[start_word]["start"]  # type: ignore[assignment]
                text_end_pos: int = (
                    word_positions[end_word - 1]["end"] if end_word > 0 else 0  # type: ignore[assignment]
                )
                chunk_text = text[text_start_pos:text_end_pos]
                chunk_words = [wp["word"] for wp in word_positions[start_word:end_word]]

                if safe_strip(chunk_text):
                    estimated_tokens = int(len(chunk_words) * approx_tokens_per_word)

                    chunk = {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "start_pos": text_start_pos,
                        "end_pos": text_end_pos,
                        "start_word": start_word,
                        "end_word": end_word,
                        "token_count_estimate": estimated_tokens,
                        "word_count": len(chunk_words),
                        "char_count": len(chunk_text),
                        **(metadata or {}),
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                # Move start from actual end to preserve overlap
                start_word = max(0, end_word - overlap_words)

        return chunks


class SemanticChunker(ChunkingStrategy):
    """Semantic boundary-aware chunking."""

    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk text based on semantic boundaries like paragraphs and sentences."""
        chunks = []
        chunk_index = 0

        # First try to split on paragraphs
        paragraphs = self.split_on_paragraphs(text)

        current_chunk = ""
        start_pos = 0

        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                trimmed_text = safe_strip(current_chunk)
                if trimmed_text:
                    chunk = {
                        "text": trimmed_text,
                        "chunk_index": chunk_index,
                        "start_pos": start_pos,
                        "end_pos": start_pos + len(trimmed_text),
                        "word_count": len(trimmed_text.split()),
                        "char_count": len(trimmed_text),
                        "token_count_estimate": self.token_counter.count_tokens(
                            trimmed_text
                        ),
                        **(metadata or {}),
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                # Capture the previous chunk length before mutation
                prev_len = len(current_chunk)

                # Start new chunk with overlap
                if self.overlap > 0:
                    # Guard overlap slicing when prev_len is smaller than configured overlap
                    overlap_text = current_chunk[-min(self.overlap, prev_len) :]
                    current_chunk = overlap_text + paragraph
                    # Compute start_pos from the captured prev_len
                    start_pos = start_pos + (prev_len - len(overlap_text))
                else:
                    current_chunk = paragraph
                    # Move start_pos by the full prev_len when no overlap
                    start_pos = start_pos + prev_len
            else:
                current_chunk += paragraph

        # Add final chunk
        trimmed_text = safe_strip(current_chunk)
        if trimmed_text:
            chunk = {
                "text": trimmed_text,
                "chunk_index": chunk_index,
                "start_pos": start_pos,
                "end_pos": start_pos + len(trimmed_text),
                "word_count": len(trimmed_text.split()),
                "char_count": len(trimmed_text),
                "token_count_estimate": self.token_counter.count_tokens(trimmed_text),
                **(metadata or {}),
            }
            chunks.append(chunk)

        return chunks

    def find_semantic_boundaries(self, text: str) -> list[int]:
        """Find semantic boundaries in text using helper functions."""
        boundaries = []
        text_length = len(text)

        # Check every N characters for potential boundaries
        step = min(100, self.chunk_size // 10)
        for pos in range(step, text_length, step):
            # Try to find a boundary near this position
            search_start = max(0, pos - 50)
            search_end = min(text_length, pos + 50)
            search_text = text[search_start:search_end]
            relative_pos = pos - search_start

            # Try paragraph boundary first
            boundary = find_paragraph_boundary(search_text, relative_pos)
            if boundary is not None:
                boundaries.append(search_start + boundary)
            elif pos % (step * 2) == 0:  # Check sentence boundaries less frequently
                boundary = find_sentence_boundary(search_text, relative_pos)
                if boundary is not None:
                    boundaries.append(search_start + boundary)

        return sorted(set(boundaries))

    def split_on_sentences(self, text: str) -> list[str]:
        """Split text on sentence boundaries."""
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+\s+", text)
        return [safe_strip(s) for s in sentences if safe_strip(s)]

    def split_on_paragraphs(self, text: str) -> list[str]:
        """Split text on paragraph boundaries."""
        paragraphs = text.split("\n\n")
        return [safe_strip(p) for p in paragraphs if safe_strip(p)]


class AdaptiveChunker(ChunkingStrategy):
    """Adaptive chunking based on content type and structure."""

    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk text using adaptive strategy based on content type."""
        content_type = self.detect_content_structure(text)

        if content_type == "code":
            return self.chunk_code_content(text, metadata)
        elif content_type == "markdown":
            return self.chunk_markdown_content(text, metadata)
        else:
            return self.chunk_plain_text(text, metadata)

    def detect_content_structure(self, text: str) -> str:
        """Detect the structure/type of content."""
        # Simple heuristics for content type detection
        lines = text.split("\n")

        # Check for code indicators
        code_indicators = 0
        markdown_indicators = 0

        for line in lines[:20]:  # Check first 20 lines
            if any(
                keyword in line
                for keyword in [
                    "def ",
                    "class ",
                    "function ",
                    "import ",
                    "const ",
                    "var ",
                ]
            ):
                code_indicators += 1
            if any(
                marker in line for marker in ["# ", "## ", "### ", "* ", "- ", "```"]
            ):
                markdown_indicators += 1

        if code_indicators > markdown_indicators:
            return "code"
        elif markdown_indicators > 0:
            return "markdown"
        else:
            return "plain_text"

    def chunk_code_content(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk code content preserving function/class boundaries."""
        # For code, try to keep functions/classes together
        lines = text.split("\n")
        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_line = 0

        # Build cumulative character position map for lines
        line_char_positions = [0]  # Start position of each line
        char_pos = 0
        for line in lines:
            char_pos += len(line) + 1  # +1 for newline
            line_char_positions.append(char_pos)

        # Track character position for current chunk start
        char_start_pos = 0

        for i, line in enumerate(lines):
            # Check if we're starting a new function/class and current chunk is getting large
            trimmed_text = safe_strip(current_chunk)
            if (
                any(keyword in line for keyword in ["def ", "class ", "function "])
                and len(current_chunk) > self.chunk_size * 0.8
                and trimmed_text
            ):
                chunk = {
                    "text": trimmed_text,
                    "chunk_index": chunk_index,
                    "start_pos": char_start_pos,  # Character offset
                    "end_pos": char_start_pos + len(trimmed_text),  # Character offset
                    "start_line": start_line,  # Line index for backward compat
                    "end_line": i,  # Line index for backward compat
                    "word_count": len(trimmed_text.split()),
                    "char_count": len(trimmed_text),
                    "token_count_estimate": self.token_counter.count_tokens(
                        trimmed_text
                    ),
                    "content_type": "code",
                    **(metadata or {}),
                }
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = ""
                start_line = i
                char_start_pos = (
                    line_char_positions[i] if i < len(line_char_positions) else char_pos
                )

            current_chunk += line + "\n"

            # Force split if chunk gets too large
            trimmed_text = safe_strip(current_chunk)
            if len(current_chunk) > self.chunk_size * 1.5 and trimmed_text:
                chunk = {
                    "text": trimmed_text,
                    "chunk_index": chunk_index,
                    "start_pos": char_start_pos,  # Character offset
                    "end_pos": char_start_pos + len(trimmed_text),  # Character offset
                    "start_line": start_line,  # Line index for backward compat
                    "end_line": i + 1,  # Line index for backward compat
                    "word_count": len(trimmed_text.split()),
                    "char_count": len(trimmed_text),
                    "token_count_estimate": self.token_counter.count_tokens(
                        trimmed_text
                    ),
                    "content_type": "code",
                    **(metadata or {}),
                }
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = ""
                start_line = i + 1
                char_start_pos = (
                    line_char_positions[min(i + 1, len(lines))]
                    if (i + 1) < len(line_char_positions)
                    else char_pos
                )

        # Add final chunk
        trimmed_text = safe_strip(current_chunk)
        if trimmed_text:
            chunk = {
                "text": trimmed_text,
                "chunk_index": chunk_index,
                "start_pos": char_start_pos,  # Character offset
                "end_pos": char_start_pos + len(trimmed_text),  # Character offset
                "start_line": start_line,  # Line index for backward compat
                "end_line": len(lines),  # Line index for backward compat
                "word_count": len(trimmed_text.split()),
                "char_count": len(trimmed_text),
                "token_count_estimate": self.token_counter.count_tokens(trimmed_text),
                "content_type": "code",
                **(metadata or {}),
            }
            chunks.append(chunk)

        return chunks

    def chunk_markdown_content(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk markdown content preserving section boundaries."""
        # Use semantic chunker for markdown but with section awareness
        semantic_chunker = SemanticChunker(self.chunk_size, self.overlap)
        chunks = semantic_chunker.chunk_text(text, metadata)

        # Add markdown-specific metadata
        for chunk in chunks:
            chunk["content_type"] = "markdown"

        return chunks

    def chunk_plain_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk plain text using fixed-size strategy."""
        fixed_chunker = FixedSizeChunker(self.chunk_size, self.overlap)
        chunks = fixed_chunker.chunk_text(text, metadata)

        # Add content type metadata
        for chunk in chunks:
            chunk["content_type"] = "plain_text"

        return chunks
