"""
Configuration management for Crawler-MCP using Pydantic Settings.
"""

import logging
import random
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    CHUNK_OVERLAP_ERROR,
    CHUNK_SIZE_ERROR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    # Content processing constants
    DEFAULT_CHUNK_SIZE,
    # Pool & concurrency constants
    DEFAULT_CONNECTION_POOL_SIZE,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MAX_LENGTH,
    DEFAULT_EMBEDDING_MAX_RETRIES,
    DEFAULT_EMBEDDING_WORKERS,
    DEFAULT_EXCLUDED_URL_PATTERNS,
    # Server & Network constants
    DEFAULT_HOST,
    DEFAULT_MAX_CONCURRENT_CRAWLS,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_PORT,
    DEFAULT_PREFETCH_SIZE,
    DEFAULT_PRUNING_THRESHOLD,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_TOP_K,
    DEFAULT_RETRY_COUNT,
    EMBEDDING_WORKERS_ERROR,
    MAX_BATCH_SIZE,
    MAX_CHUNK_SIZE,
    MAX_CONNECTION_POOL_SIZE,
    MAX_EMBEDDING_WORKERS,
    MAX_MAX_FILE_SIZE_MB,
    MAX_PREFETCH_SIZE,
    MAX_REQUEST_SIZE_BYTES,
    MAX_RERANKER_MAX_LENGTH,
    MAX_RERANKER_TOP_K,
    MAX_RETRY_EXPONENTIAL_BASE,
    MAX_RETRY_INITIAL_DELAY,
    MAX_RETRY_MAX_DELAY,
    MAX_TEI_TOKENS_PER_ITEM,
    MIN_BATCH_SIZE,
    MIN_CHUNK_SIZE,
    MIN_CONNECTION_POOL_SIZE,
    MIN_EMBEDDING_WORKERS,
    MIN_MAX_FILE_SIZE_MB,
    MIN_PREFETCH_SIZE,
    MIN_RERANKER_MAX_LENGTH,
    MIN_RERANKER_TOP_K,
    MIN_RETRY_EXPONENTIAL_BASE,
    # Validation ranges
    MIN_RETRY_INITIAL_DELAY,
    MIN_RETRY_MAX_DELAY,
    MIN_TEI_TOKENS_PER_ITEM,
    OAUTH_BASE_URL_PROD_ERROR,
    QDRANT_DEFAULT_BATCH_SIZE,
    QDRANT_DEFAULT_TIMEOUT,
    QDRANT_MAX_BATCH_SIZE,
    QDRANT_MIN_BATCH_SIZE,
    # Error messages
    RERANKER_MODEL_ERROR,
    RETRY_EXPONENTIAL_BASE,
    RETRY_INITIAL_DELAY,
    RETRY_MAX_DELAY,
    TEI_DEFAULT_BATCH_SIZE,
    TEI_DEFAULT_TIMEOUT,
    TEI_MAX_BATCH_TOKENS,
    TEI_MAX_CONCURRENT_REQUESTS,
    TEI_TOKENS_PER_ITEM,
    DeduplicationStrategy,
    VectorDistance,
)


class ConfigurationError(Exception):
    """Configuration-related error exception."""

    pass


class CrawlerMCPSettings(BaseSettings):
    """
    Application settings loaded from environment variables and .env files.
    """

    # Server Configuration
    server_host: str = Field(default=DEFAULT_HOST, alias="SERVER_HOST")
    server_port: int = Field(default=DEFAULT_PORT, alias="SERVER_PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    production: bool = Field(default=False, alias="PRODUCTION")

    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="console", alias="LOG_FORMAT")
    log_file: str | None = Field(default=None, alias="LOG_FILE")
    log_to_file: bool = Field(default=False, alias="LOG_TO_FILE")
    pid_file: str = Field(default="logs/crawlerr.pid", alias="PID_FILE")

    # Uvicorn/ASGI logging toggles (for connection debugging)
    uvicorn_access_log: bool = Field(default=True, alias="UVICORN_ACCESS_LOG")
    uvicorn_log_level: Literal[
        "critical", "error", "warning", "info", "debug", "trace"
    ] = Field(
        default="info",
        alias="UVICORN_LOG_LEVEL",
        description="Uvicorn log level (critical, error, warning, info, debug, trace)",
    )
    request_log: bool = Field(
        default=False,
        alias="REQUEST_LOG",
        description="Log each incoming HTTP/WS request at info level",
    )

    # Qdrant Vector Database
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        default="crawlerr_documents", alias="QDRANT_COLLECTION"
    )
    qdrant_vector_size: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSION, alias="QDRANT_VECTOR_SIZE"
    )
    qdrant_distance: VectorDistance = Field(
        default=VectorDistance.COSINE, alias="QDRANT_DISTANCE"
    )
    qdrant_timeout: float = Field(
        default=QDRANT_DEFAULT_TIMEOUT, alias="QDRANT_TIMEOUT"
    )
    qdrant_retry_count: int = Field(
        default=DEFAULT_RETRY_COUNT, alias="QDRANT_RETRY_COUNT"
    )
    qdrant_connection_pool_size: int = Field(
        default=DEFAULT_CONNECTION_POOL_SIZE,
        alias="QDRANT_CONNECTION_POOL_SIZE",
        ge=MIN_CONNECTION_POOL_SIZE,
        le=MAX_CONNECTION_POOL_SIZE,
    )
    # Unified batch configuration for optimal performance
    default_batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        alias="DEFAULT_BATCH_SIZE",
        ge=MIN_BATCH_SIZE,
        le=MAX_BATCH_SIZE,
        description="Default batch size for all operations",
    )
    qdrant_batch_size: int = Field(
        default=QDRANT_DEFAULT_BATCH_SIZE,
        alias="QDRANT_BATCH_SIZE",
        ge=QDRANT_MIN_BATCH_SIZE,
        le=QDRANT_MAX_BATCH_SIZE,
    )
    qdrant_prefetch_size: int = Field(
        default=DEFAULT_PREFETCH_SIZE,
        alias="QDRANT_PREFETCH_SIZE",
        ge=MIN_PREFETCH_SIZE,
        le=MAX_PREFETCH_SIZE,
    )
    qdrant_search_exact: bool = Field(default=False, alias="QDRANT_SEARCH_EXACT")

    # Vector Service Configuration - using modular implementation

    # HF Text Embeddings Inference (TEI)
    tei_url: str = Field(default="http://localhost:8080", alias="TEI_URL")
    tei_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B", alias="TEI_MODEL")
    tei_max_concurrent_requests: int = Field(
        default=TEI_MAX_CONCURRENT_REQUESTS,
        alias="TEI_MAX_CONCURRENT_REQUESTS",
    )
    tei_max_batch_tokens: int = Field(
        default=TEI_MAX_BATCH_TOKENS, alias="TEI_MAX_BATCH_TOKENS"
    )
    tei_tokens_per_item: int = Field(
        default=TEI_TOKENS_PER_ITEM,
        alias="TEI_TOKENS_PER_ITEM",
        ge=MIN_TEI_TOKENS_PER_ITEM,
        le=MAX_TEI_TOKENS_PER_ITEM,
        description="Estimated tokens per embedding item for batch size calculation",
    )
    tei_batch_size: int = Field(
        default=TEI_DEFAULT_BATCH_SIZE,
        alias="TEI_BATCH_SIZE",
    )  # Will be validated against TEI_MAX_BATCH_TOKENS
    tei_timeout: float = Field(default=TEI_DEFAULT_TIMEOUT, alias="TEI_TIMEOUT")

    # Embedding Configuration
    embedding_max_length: int = Field(
        default=DEFAULT_EMBEDDING_MAX_LENGTH, alias="EMBEDDING_MAX_LENGTH"
    )
    embedding_dimension: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSION, alias="EMBEDDING_DIMENSION"
    )
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    embedding_max_retries: int = Field(
        default=DEFAULT_EMBEDDING_MAX_RETRIES, alias="EMBEDDING_MAX_RETRIES"
    )

    # Retry configuration with exponential backoff
    retry_initial_delay: float = Field(
        default=RETRY_INITIAL_DELAY,
        alias="RETRY_INITIAL_DELAY",
        ge=MIN_RETRY_INITIAL_DELAY,
        le=MAX_RETRY_INITIAL_DELAY,
        description="Initial delay in seconds for exponential backoff",
    )
    retry_max_delay: float = Field(
        default=RETRY_MAX_DELAY,
        alias="RETRY_MAX_DELAY",
        ge=MIN_RETRY_MAX_DELAY,
        le=MAX_RETRY_MAX_DELAY,
        description="Maximum delay in seconds for exponential backoff",
    )
    retry_exponential_base: float = Field(
        default=RETRY_EXPONENTIAL_BASE,
        alias="RETRY_EXPONENTIAL_BASE",
        ge=MIN_RETRY_EXPONENTIAL_BASE,
        le=MAX_RETRY_EXPONENTIAL_BASE,
        description="Base for exponential backoff calculation",
    )
    embedding_workers: int = Field(
        default=DEFAULT_EMBEDDING_WORKERS,
        alias="EMBEDDING_WORKERS",
        ge=MIN_EMBEDDING_WORKERS,
        le=MAX_EMBEDDING_WORKERS,
    )

    # Chunking Configuration
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        alias="CHUNK_SIZE",
        gt=MIN_CHUNK_SIZE,
        le=MAX_CHUNK_SIZE,
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP, alias="CHUNK_OVERLAP", ge=0
    )
    word_to_token_ratio: float = Field(default=1.4, alias="WORD_TO_TOKEN_RATIO", gt=0)

    # Reranker Configuration
    reranker_model: str = Field(
        default="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", alias="RERANKER_MODEL"
    )
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_top_k: int = Field(
        default=DEFAULT_RERANKER_TOP_K,
        alias="RERANKER_TOP_K",
        gt=MIN_RERANKER_TOP_K,
        le=MAX_RERANKER_TOP_K,
    )
    reranker_max_length: int = Field(
        default=DEFAULT_RERANKER_MAX_LENGTH,
        alias="RERANKER_MAX_LENGTH",
        gt=MIN_RERANKER_MAX_LENGTH,
        le=MAX_RERANKER_MAX_LENGTH,
    )
    reranker_fallback_to_custom: bool = Field(
        default=True, alias="RERANKER_FALLBACK_TO_CUSTOM"
    )

    @field_validator("reranker_model", mode="before")
    @classmethod
    def _validate_reranker_model(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError(RERANKER_MODEL_ERROR)
        return v

    @field_validator("embedding_workers")
    @classmethod
    def validate_embedding_workers(cls, v: int) -> int:
        if not MIN_EMBEDDING_WORKERS <= v <= MAX_EMBEDDING_WORKERS:
            raise ValueError(EMBEDDING_WORKERS_ERROR)
        return v

    @field_validator("tei_url", "qdrant_url")
    @classmethod
    def validate_service_url(cls, v: str) -> str:
        """Validate service URLs have proper format."""
        if not v:
            raise ValueError("Service URL cannot be empty")

        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(
                    f"Invalid URL format: {v}. Must include scheme and host."
                )
            if parsed.scheme not in ("http", "https"):
                raise ValueError(
                    f"URL scheme must be http or https, got: {parsed.scheme}"
                )
        except Exception as e:
            raise ValueError(f"Invalid URL: {e}") from e

        return v

    @field_validator("qdrant_distance")
    @classmethod
    def validate_qdrant_distance(cls, v: VectorDistance) -> VectorDistance:
        """Validate Qdrant distance metric."""
        if v not in VectorDistance:
            valid_values = [d.value for d in VectorDistance]
            raise ValueError(
                f"Invalid distance metric: {v}. Must be one of: {valid_values}"
            )
        return v

    @field_validator("deduplication_strategy")
    @classmethod
    def validate_deduplication_strategy(
        cls, v: DeduplicationStrategy
    ) -> DeduplicationStrategy:
        """Validate deduplication strategy."""
        if v not in DeduplicationStrategy:
            valid_values = [s.value for s in DeduplicationStrategy]
            raise ValueError(
                f"Invalid deduplication strategy: {v}. Must be one of: {valid_values}"
            )
        return v

    # GPU flag validation removed - no longer using custom Chrome flags

    # Crawling Configuration
    max_concurrent_crawls: int = Field(
        default=DEFAULT_MAX_CONCURRENT_CRAWLS, alias="MAX_CONCURRENT_CRAWLS"
    )
    # Deprecated: max_crawl_pages removed; use MAX_PAGES via CrawlerSettings

    # Streaming Configuration - Single source of truth
    enable_streaming: bool = Field(
        default=True,
        alias="ENABLE_STREAMING",
        description="Enable streaming mode for real-time processing and reduced memory usage",
    )

    # URL Pattern Exclusions - Conservative defaults to avoid admin/auth areas and binary files
    crawl_exclude_url_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_URL_PATTERNS),
        alias="CRAWL_EXCLUDE_URL_PATTERNS",
        description="URL patterns to exclude during crawling - includes admin/auth paths and binary files (override via env var for broader crawling)",
    )

    # Content Filtering Configuration - Clean Markdown Generation
    crawl_excluded_tags: list[str] = Field(
        default=[
            "script",
            "style",
        ],
        alias="CRAWL_EXCLUDED_TAGS",
        description="HTML tags to exclude during content extraction (optimized for crawl4ai - only exclude script/style)",
    )

    crawl_strict_ui_filtering: bool = Field(
        default=False,
        alias="CRAWL_STRICT_UI_FILTERING",
        description="When true, also exclude alerts/notifications and other aggressive UI elements that may contain documentation content",
    )

    @property
    def crawl_excluded_selectors_list(self) -> list[str]:
        """Get excluded selectors list based on strict filtering setting."""
        base_selectors = [
            # Copy buttons - comprehensive patterns
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
            # Tab navigation - all variants
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
            # Navigation elements
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
            # Documentation UI artifacts (safe to always exclude)
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
            # Search and interactive elements
            ".search-box",
            ".filter-bar",
            ".sort-options",
            ".search-input",
        ]

        # Add aggressive selectors only if strict filtering is enabled
        if self.crawl_strict_ui_filtering:
            base_selectors.extend(
                [
                    ".alert",
                    ".notification",
                ]
            )

        return base_selectors

    crawl_excluded_selectors: list[str] = Field(
        default_factory=list,  # Will be populated by property
        alias="CRAWL_EXCLUDED_SELECTORS",
        description="CSS selectors for UI elements to exclude from content extraction (use crawl_strict_ui_filtering for alerts/notifications)",
    )

    crawl_content_selector: str | None = Field(
        default=None,
        alias="CRAWL_CONTENT_SELECTOR",
        description="CSS selector to focus on main content area (None = no CSS filtering; crawler should not auto-inject)",
    )

    crawl_use_semantic_default_selector: bool = Field(
        default=False,
        alias="CRAWL_USE_SEMANTIC_DEFAULT_SELECTOR",
        description="When true and content_selector is None, automatically apply semantic HTML5 selectors (main, article, [role=main])",
    )

    crawl_pruning_threshold: float = Field(
        default=DEFAULT_PRUNING_THRESHOLD,
        alias="CRAWL_PRUNING_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="Threshold for content relevance in PruningContentFilter (0.25 = keep 75% of content, optimized for crawl4ai)",
    )

    crawl_min_word_threshold: int = Field(
        default=3,
        alias="CRAWL_MIN_WORD_THRESHOLD",
        ge=3,
        le=100,
        description="Minimum words required for content blocks to be included (3 = optimized for crawl4ai content filtering)",
    )

    crawl_prefer_fit_markdown: bool = Field(
        default=True,
        alias="CRAWL_PREFER_FIT_MARKDOWN",
        description="Prefer fit_markdown over raw_markdown for cleaner content",
    )

    clean_ui_artifacts: bool = Field(
        default=True,
        alias="CLEAN_UI_ARTIFACTS",
        description="Enable post-processing regex cleanup of UI artifacts like Copy buttons and tab navigation",
    )

    # Deduplication Configuration
    deduplication_enabled: bool = Field(
        default=True,
        alias="DEDUPLICATION_ENABLED",
        description="Enable content-based deduplication for crawled pages",
    )
    deduplication_strategy: DeduplicationStrategy = Field(
        default=DeduplicationStrategy.CONTENT_HASH,
        alias="DEDUPLICATION_STRATEGY",
        description="Deduplication strategy: 'content_hash', 'timestamp', or 'none'",
    )

    # Directory Crawling Configuration
    directory_excluded_extensions: list[str] = Field(
        default=[
            # Binary executables and libraries
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            ".obj",
            ".o",
            # Images
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".ico",
            ".tiff",
            ".webp",
            ".svg",
            # Audio/Video
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".wav",
            ".mkv",
            ".webm",
            # Archives
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".7z",
            ".rar",
            ".xz",
            # Documents (can be made configurable for future PDF/Office extraction)
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            # Other binary formats
            ".iso",
            ".dmg",
            ".pkg",
            ".deb",
            ".rpm",
        ],
        alias="DIRECTORY_EXCLUDED_EXTENSIONS",
        description="File extensions to exclude when crawling directories",
    )
    directory_max_file_size_mb: int = Field(
        default=DEFAULT_MAX_FILE_SIZE_MB,
        alias="DIRECTORY_MAX_FILE_SIZE_MB",
        ge=MIN_MAX_FILE_SIZE_MB,
        le=MAX_MAX_FILE_SIZE_MB,
        description="Maximum file size in MB to process when crawling directories",
    )

    # OAuth Configuration
    oauth_enabled: bool = Field(default=False, alias="OAUTH_ENABLED")
    oauth_provider: str | None = Field(default=None, alias="OAUTH_PROVIDER")

    # Google OAuth Settings
    google_client_id: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_ID"
    )
    google_client_secret: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_SECRET"
    )
    google_base_url: str | None = Field(
        default=None, alias="FASTMCP_SERVER_AUTH_GOOGLE_BASE_URL"
    )
    google_required_scopes: str = Field(
        default="openid,email,profile",
        alias="FASTMCP_SERVER_AUTH_GOOGLE_REQUIRED_SCOPES",
    )

    # CORS & Security
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    cors_credentials: bool = Field(default=True, alias="CORS_CREDENTIALS")
    max_request_size: int = Field(
        default=MAX_REQUEST_SIZE_BYTES, alias="MAX_REQUEST_SIZE"
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT, alias="REQUEST_TIMEOUT"
    )

    def compute_retry_backoff(self, attempts: int) -> float:
        """Compute exponential backoff delay with jitter."""
        base_delay = min(
            self.retry_max_delay,
            self.retry_initial_delay * (self.retry_exponential_base**attempts),
        )
        # Apply jitter (±20%)
        jittered_delay = base_delay * random.uniform(0.8, 1.2)
        return max(self.retry_initial_delay, min(self.retry_max_delay, jittered_delay))

    @property
    def cors_origins_list(self) -> list[str]:
        """Convert cors_origins string to list."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    @property
    def google_scopes_list(self) -> list[str]:
        """Convert google_required_scopes string to list."""
        return [
            scope.strip()
            for scope in self.google_required_scopes.split(",")
            if scope.strip()
        ]

    @field_validator("log_file", mode="before")
    @classmethod
    def create_log_directory(cls, v: str | None) -> str | None:
        if v:
            log_path = Path(str(v)).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("pid_file", mode="before")
    @classmethod
    def create_pid_directory(cls, v: str) -> str:
        pid_path = Path(str(v)).expanduser()
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def _validate_chunking(self) -> "CrawlerMCPSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(CHUNK_OVERLAP_ERROR)
        if self.chunk_size > self.embedding_max_length:
            raise ValueError(CHUNK_SIZE_ERROR)
        return self

    @model_validator(mode="after")
    def _validate_tei_batch_size(self) -> "CrawlerMCPSettings":
        """Validate TEI batch size against token limits."""
        estimated_tokens = self.tei_batch_size * self.tei_tokens_per_item
        if estimated_tokens > self.tei_max_batch_tokens:
            import logging

            logger = logging.getLogger(__name__)
            derived_batch_size = max(
                1, self.tei_max_batch_tokens // self.tei_tokens_per_item
            )
            logger.warning(
                "TEI batch size %s * %s tokens/item = %s exceeds TEI_MAX_BATCH_TOKENS %s. "
                "Consider reducing to %s for optimal performance.",
                self.tei_batch_size,
                self.tei_tokens_per_item,
                estimated_tokens,
                self.tei_max_batch_tokens,
                derived_batch_size,
            )
        return self

    @model_validator(mode="after")
    def _validate_batch_consistency(self) -> "CrawlerMCPSettings":
        """Validate batch size consistency across services."""
        # Ensure vector dimension consistency
        if self.qdrant_vector_size != self.embedding_dimension:
            logging.getLogger(__name__).warning(
                "QDRANT_VECTOR_SIZE (%s) differs from EMBEDDING_DIMENSION (%s). "
                "This may cause vector storage issues.",
                self.qdrant_vector_size,
                self.embedding_dimension,
            )

        # Validate batch sizes are reasonable
        if self.default_batch_size != self.qdrant_batch_size:
            logging.getLogger(__name__).info(
                "Different batch sizes configured: DEFAULT_BATCH_SIZE=%s, QDRANT_BATCH_SIZE=%s",
                self.default_batch_size,
                self.qdrant_batch_size,
            )

        return self

    @model_validator(mode="after")
    def _populate_excluded_selectors(self) -> "CrawlerMCPSettings":
        """Populate crawl_excluded_selectors from property if it's empty."""
        if not self.crawl_excluded_selectors:
            self.crawl_excluded_selectors = self.crawl_excluded_selectors_list
        return self

    @model_validator(mode="after")
    def _validate_oauth(self) -> "CrawlerMCPSettings":
        """Auto-enable OAuth if Google credentials are provided and validate configuration."""
        if self.google_client_id and self.google_client_secret:
            self.oauth_enabled = True
            self.oauth_provider = "google"

            if not self.google_base_url:
                if self.production:
                    # In production, require explicit base URL configuration
                    raise ConfigurationError(OAUTH_BASE_URL_PROD_ERROR)
                else:
                    # In development/test, auto-derive base URL as fallback
                    protocol = "https" if self.production else "http"
                    self.google_base_url = (
                        f"{protocol}://{self.server_host}:{self.server_port}"
                    )
        return self

    @model_validator(mode="after")
    def _validate_service_compatibility(self) -> "CrawlerMCPSettings":
        """Validate compatibility between services and configurations."""
        logger = logging.getLogger(__name__)

        # Check embedding service vs Qdrant compatibility
        if self.embedding_dimension != self.qdrant_vector_size:
            logger.warning(
                "Embedding dimension (%s) differs from Qdrant vector size (%s). "
                "This may cause vector storage issues.",
                self.embedding_dimension,
                self.qdrant_vector_size,
            )

        # Validate OAuth configuration completeness
        if self.oauth_enabled and (
            not self.google_client_id or not self.google_client_secret
        ):
            raise ValueError(
                "OAuth is enabled but Google client credentials are missing. "
                "Please set FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_ID and "
                "FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_SECRET"
            )

        # Check reranker configuration
        if self.reranker_enabled and not self.reranker_model:
            raise ValueError("Reranker is enabled but no reranker model is specified")

        # Validate embedding workers vs concurrent crawls ratio
        if self.embedding_workers > self.max_concurrent_crawls:
            logger.info(
                "Embedding workers (%s) exceed concurrent crawls (%s). "
                "Consider balancing for optimal resource usage.",
                self.embedding_workers,
                self.max_concurrent_crawls,
            )

        return self

    model_config = SettingsConfigDict(
        env_file=(
            # Prefer package-level .env; also read legacy optimized/.env if present
            Path(__file__).parent / ".env",
            Path(__file__).parent / "crawlers" / "optimized" / ".env",
            Path(__file__).parent.parent / ".env",  # Project root .env
            ".env",  # Current directory .env as fallback
        ),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore deprecated settings for backward compatibility
    )


# Lazy settings accessor (avoids import-time side effects)
_settings: CrawlerMCPSettings | None = None


def get_settings() -> CrawlerMCPSettings:
    global _settings
    if _settings is None:
        _settings = CrawlerMCPSettings()
    return _settings


# For backward compatibility
settings = get_settings()
