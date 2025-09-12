"""
Configuration constants for the Crawler MCP server.

This module centralizes all magic numbers, default values, and string enumerations
used throughout the crawler configuration system to improve maintainability.
"""

from enum import Enum
from typing import Final

# =============================================================================
# SERVER & NETWORK CONSTANTS
# =============================================================================

# Default server configuration
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8000

# Network timeouts (in seconds)
DEFAULT_TIMEOUT: Final[float] = 30.0
QDRANT_DEFAULT_TIMEOUT: Final[float] = 10.0
TEI_DEFAULT_TIMEOUT: Final[float] = 30.0

# Retry configuration
DEFAULT_RETRY_COUNT: Final[int] = 3
RETRY_INITIAL_DELAY: Final[float] = 1.0
RETRY_MAX_DELAY: Final[float] = 60.0
RETRY_EXPONENTIAL_BASE: Final[float] = 2.0

# Request limits
MAX_REQUEST_SIZE_BYTES: Final[int] = 10485760  # 10MB
DEFAULT_REQUEST_TIMEOUT: Final[float] = 30.0

# =============================================================================
# BATCH PROCESSING CONSTANTS
# =============================================================================

# Batch sizes for various operations
DEFAULT_BATCH_SIZE: Final[int] = 256
MIN_BATCH_SIZE: Final[int] = 64
MAX_BATCH_SIZE: Final[int] = 512

# Qdrant specific batch settings
QDRANT_DEFAULT_BATCH_SIZE: Final[int] = 256
QDRANT_MIN_BATCH_SIZE: Final[int] = 64
QDRANT_MAX_BATCH_SIZE: Final[int] = 512

# TEI batch settings
TEI_DEFAULT_BATCH_SIZE: Final[int] = 512
TEI_MAX_BATCH_TOKENS: Final[int] = 32768
TEI_TOKENS_PER_ITEM: Final[int] = 30

# Prefetch sizes
DEFAULT_PREFETCH_SIZE: Final[int] = 2048
MIN_PREFETCH_SIZE: Final[int] = 256
MAX_PREFETCH_SIZE: Final[int] = 4096

# =============================================================================
# POOL & CONCURRENCY CONSTANTS
# =============================================================================

# Connection pools
DEFAULT_CONNECTION_POOL_SIZE: Final[int] = 32
MIN_CONNECTION_POOL_SIZE: Final[int] = 1
MAX_CONNECTION_POOL_SIZE: Final[int] = 64

# Worker configurations
DEFAULT_EMBEDDING_WORKERS: Final[int] = 8
MIN_EMBEDDING_WORKERS: Final[int] = 1
MAX_EMBEDDING_WORKERS: Final[int] = 16

# Concurrent processing
DEFAULT_MAX_CONCURRENT_CRAWLS: Final[int] = 25
MIN_CONCURRENT_CRAWLS: Final[int] = 1
MAX_CONCURRENT_CRAWLS: Final[int] = 50

# Crawl limits
DEFAULT_MAX_CRAWL_PAGES: Final[int] = 100

# TEI concurrency
TEI_MAX_CONCURRENT_REQUESTS: Final[int] = 256

# =============================================================================
# MEMORY MANAGEMENT CONSTANTS
# =============================================================================

# Memory thresholds (percentage)
DEFAULT_MEMORY_THRESHOLD: Final[float] = 70.0
AGGRESSIVE_MEMORY_THRESHOLD: Final[float] = 85.0
CONSERVATIVE_MEMORY_THRESHOLD: Final[float] = 60.0
MIN_MEMORY_THRESHOLD: Final[float] = 10.0
MAX_MEMORY_THRESHOLD: Final[float] = 95.0

# =============================================================================
# CONTENT PROCESSING CONSTANTS
# =============================================================================

# Chunking configuration
DEFAULT_CHUNK_SIZE: Final[int] = 1024
MIN_CHUNK_SIZE: Final[int] = 100
MAX_CHUNK_SIZE: Final[int] = 32768
DEFAULT_CHUNK_OVERLAP: Final[int] = 200

# Content thresholds
DEFAULT_MIN_WORD_COUNT: Final[int] = 50
DEFAULT_PRUNING_THRESHOLD: Final[float] = 0.25
DEFAULT_CONTENT_THRESHOLD: Final[float] = 0.48

# Embedding configuration
DEFAULT_EMBEDDING_MAX_LENGTH: Final[int] = 32000
DEFAULT_EMBEDDING_DIMENSION: Final[int] = 1024
DEFAULT_EMBEDDING_MAX_RETRIES: Final[int] = 3

# Reranker configuration
DEFAULT_RERANKER_TOP_K: Final[int] = 10
MIN_RERANKER_TOP_K: Final[int] = 1
MAX_RERANKER_TOP_K: Final[int] = 100
DEFAULT_RERANKER_MAX_LENGTH: Final[int] = 512
MIN_RERANKER_MAX_LENGTH: Final[int] = 1
MAX_RERANKER_MAX_LENGTH: Final[int] = 4096

# =============================================================================
# VIEWPORT & BROWSER CONSTANTS
# =============================================================================

# Viewport dimensions
DEFAULT_VIEWPORT_WIDTH: Final[int] = 1280
DEFAULT_VIEWPORT_HEIGHT: Final[int] = 720
MINIMAL_VIEWPORT_WIDTH: Final[int] = 800
MINIMAL_VIEWPORT_HEIGHT: Final[int] = 600

# Browser timeouts (milliseconds)
DEFAULT_PAGE_TIMEOUT: Final[int] = 30000
AGGRESSIVE_PAGE_TIMEOUT: Final[int] = 15000
CONSERVATIVE_PAGE_TIMEOUT: Final[int] = 60000

# =============================================================================
# FILE & DIRECTORY CONSTANTS
# =============================================================================

# File size limits
DEFAULT_MAX_FILE_SIZE_MB: Final[int] = 10
MIN_MAX_FILE_SIZE_MB: Final[int] = 1
MAX_MAX_FILE_SIZE_MB: Final[int] = 100

# Log rotation
DEFAULT_LOG_ROTATION_SIZE_MB: Final[int] = 10
DEFAULT_LOG_ROTATION_BACKUPS: Final[int] = 3

# =============================================================================
# VALIDATION RANGES
# =============================================================================

# Retry validation ranges
MIN_RETRY_INITIAL_DELAY: Final[float] = 0.1
MAX_RETRY_INITIAL_DELAY: Final[float] = 10.0
MIN_RETRY_MAX_DELAY: Final[float] = 1.0
MAX_RETRY_MAX_DELAY: Final[float] = 300.0
MIN_RETRY_EXPONENTIAL_BASE: Final[float] = 1.1
MAX_RETRY_EXPONENTIAL_BASE: Final[float] = 5.0

# TEI validation ranges
MIN_TEI_TOKENS_PER_ITEM: Final[int] = 10
MAX_TEI_TOKENS_PER_ITEM: Final[int] = 100

# =============================================================================
# STRING ENUMERATIONS
# =============================================================================


class VectorDistance(str, Enum):
    """Vector distance metrics for similarity search."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


class DeduplicationStrategy(str, Enum):
    """Content deduplication strategies."""

    CONTENT_HASH = "content_hash"
    TIMESTAMP = "timestamp"
    NONE = "none"


class BrowserMode(str, Enum):
    """Browser operation modes for crawling.

    Note: Added HEADLESS to match settings usage.
    """

    FULL = "full"
    TEXT = "text"
    MINIMAL = "minimal"
    HEADLESS = "headless"


class CacheStrategy(str, Enum):
    """Caching strategies for crawl operations."""

    ENABLED = "enabled"
    BYPASS = "bypass"
    DISABLED = "disabled"
    ADAPTIVE = "adaptive"


class ContentFilterType(str, Enum):
    """Content filtering algorithms."""

    PRUNING = "pruning"
    BM25 = "bm25"
    NONE = "none"


class WaitCondition(str, Enum):
    """Page load completion conditions."""

    DOM_CONTENT_LOADED = "domcontentloaded"
    NETWORK_IDLE = "networkidle"


class LogLevel(str, Enum):
    """Logging levels."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    TRACE = "trace"


class LogFormat(str, Enum):
    """Log output formats."""

    CONSOLE = "console"
    JSON = "json"


class MonitorMode(str, Enum):
    """Crawler monitoring detail levels."""

    DETAILED = "DETAILED"
    AGGREGATED = "AGGREGATED"


class PruningThresholdType(str, Enum):
    """Pruning threshold calculation methods."""

    FIXED = "fixed"
    DYNAMIC = "dynamic"


class EmbeddingProjection(str, Enum):
    """Embedding dimension projection methods."""

    NONE = "none"
    TRUNCATE = "truncate"
    PAD_ZERO = "pad_zero"


class ProxyType(str, Enum):
    """Supported proxy types for HTTP requests."""

    HTTP = "http"
    HTTPS = "https"
    SOCKS5 = "socks5"
    DIRECT = "direct"


class ExtractionStrategy(str, Enum):
    """Extraction strategies supported by the crawler.

    These are high-level strategy names used by configuration. They map onto
    concrete implementations (e.g., CSS/LLM/Cosine) in the crawl layer.
    """

    BASIC = "basic"
    CSS = "css"
    COSINE = "cosine"
    LLM = "llm"


# =============================================================================
# DEFAULT COLLECTIONS AND LISTS
# =============================================================================

# Default browser arguments for optimal performance
DEFAULT_BROWSER_ARGS: Final[tuple[str, ...]] = (
    "--disable-dev-shm-usage",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-features=TranslateUI",
)

# Default excluded HTML tags for content extraction
DEFAULT_EXCLUDED_TAGS: Final[tuple[str, ...]] = (
    "script",
    "style",
    "noscript",
    "iframe",
    "nav",
    "header",
    "footer",
    "aside",
)

# Image file extensions for validation
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".bmp",
        ".tiff",
        ".ico",
    }
)

# Binary file extensions to exclude from directory crawling
BINARY_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        # Executables and libraries
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".obj",
        ".o",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".xz",
        # Media files
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".wav",
        ".mkv",
        ".webm",
        # Documents
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        # System files
        ".iso",
        ".dmg",
        ".pkg",
        ".deb",
        ".rpm",
    }
)

# Default URL patterns to exclude during crawling
DEFAULT_EXCLUDED_URL_PATTERNS: Final[tuple[str, ...]] = (
    # Internal CDN/routing paths
    r".*/_sites/.*",
    # Fragment identifiers (anchors)
    r".*#.*",
    # Auto-generated object pages (typically anchors turned paths)
    r".*__.*__.*",  # double-underscore patterns like __init__
    r".*__init__.*",
    # Admin and authentication endpoints
    r".*/admin.*",
    r".*/login.*",
    r".*/logout.*",
    r".*/signup.*",
    r".*/auth.*",
    r".*/wp-admin.*",
    r".*/dashboard.*",
    r".*/account.*",
    r".*/profile.*",
    # Binary file extensions
    r".*\.zip$",
    r".*\.exe$",
    r".*\.bin$",
    r".*\.pdf$",
    r".*\.jpg$",
    r".*\.jpeg$",
    r".*\.png$",
    r".*\.gif$",
    r".*\.mp4$",
    r".*\.mp3$",
    r".*\.avi$",
    r".*\.mkv$",
    r".*\.iso$",
    r".*\.dmg$",
)

# Default CSS selectors to exclude from content extraction
DEFAULT_EXCLUDED_SELECTORS: Final[tuple[str, ...]] = (
    # Copy buttons
    ".copy-button",
    ".copy-code-button",
    ".copy-btn",
    ".btn-copy",
    ".btn-clipboard",
    "button[title*='Copy']",
    "button[aria-label*='Copy']",
    "button[class*='copy']",
    "button[data-copy]",
    "[data-copy-button]",
    ".clipboard-button",
    # Navigation
    ".tab-nav",
    ".tab-nav-item",
    ".tab-switcher",
    ".tabs",
    ".tab-buttons",
    ".tab-container",
    ".package-manager-tabs",
    ".code-tabs",
    "[role='tablist']",
    ".tab-list",
    "[data-tabs]",
    ".breadcrumb",
    ".breadcrumbs",
    ".nav-breadcrumb",
    ".breadcrumb-nav",
    ".sidebar",
    ".navigation",
    ".nav-menu",
    ".menu-nav",
    ".site-nav",
    ".toc-sidebar",
    ".doc-nav",
    ".header-nav",
    ".footer-nav",
    ".pagination-nav",
    ".mobile-nav",
    ".nav-toggle",
    ".hamburger-menu",
    # UI artifacts
    ".social-share",
    ".share-buttons",
    ".ad-banner",
    ".promo",
    ".banner",
    ".edit-page",
    ".improve-page",
    ".feedback",
    ".edit-link",
    ".improve-doc",
    ".report-issue",
    ".last-updated",
    ".contributors",
    ".page-metadata",
    ".version-selector",
    ".language-selector",
    ".search-box",
    ".filter-bar",
    ".sort-options",
    ".search-input",
    # Ads and tracking
    "#ads",
    ".advertisement",
    ".tracking",
    ".cookie-banner",
    ".newsletter-signup",
    "[class*='ad-']",
    "[id*='ad-']",
    ".sidebar-ad",
    ".sponsored",
)

# Default documentation URL patterns
DEFAULT_DOCUMENTATION_PATTERNS: Final[tuple[str, ...]] = (
    "*/docs/*",
    "*/documentation/*",
    "*/api/*",
    "*/guide/*",
    "*/tutorial/*",
    "*/reference/*",
    "*/manual/*",
)

# Default allowed locales for content filtering
DEFAULT_ALLOWED_LOCALES: Final[tuple[str, ...]] = ("en",)

# =============================================================================
# ERROR MESSAGES
# =============================================================================

# Validation error messages
RERANKER_MODEL_ERROR: Final[str] = "RERANKER_MODEL must be a non-empty string"
CHUNK_OVERLAP_ERROR: Final[str] = "CHUNK_OVERLAP must be less than CHUNK_SIZE"
CHUNK_SIZE_ERROR: Final[str] = "CHUNK_SIZE must be <= EMBEDDING_MAX_LENGTH"
EMBEDDING_WORKERS_ERROR: Final[str] = "embedding_workers must be between 1 and 16"
MEMORY_THRESHOLD_ERROR: Final[str] = "memory_threshold must be between 10.0 and 95.0"
DEDUPLICATION_STRATEGY_ERROR: Final[str] = (
    "DEDUPLICATION_STRATEGY must be one of: content_hash, timestamp, none"
)

# OAuth error messages
OAUTH_BASE_URL_PROD_ERROR: Final[str] = (
    "FASTMCP_SERVER_AUTH_GOOGLE_BASE_URL is required when running in production. "
    "Auto-derivation of OAuth base URLs can produce incorrect URLs behind "
    "proxies/load-balancers. Please set this environment variable to your "
    "publicly accessible server URL (e.g., https://your-domain.com)."
)
