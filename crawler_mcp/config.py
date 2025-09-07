"""
Configuration management for Crawler-MCP using Pydantic Settings.
"""

import logging
import random
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Constants for error messages
RERANKER_MODEL_ERROR = "RERANKER_MODEL must be a non-empty string"


class CrawlerMCPSettings(BaseSettings):
    """
    Application settings loaded from environment variables and .env files.
    """

    # Server Configuration
    server_host: str = Field(default="127.0.0.1", alias="SERVER_HOST")
    server_port: int = Field(default=8000, alias="SERVER_PORT")
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
    qdrant_vector_size: int = Field(default=1024, alias="QDRANT_VECTOR_SIZE")
    qdrant_distance: Literal["cosine", "euclidean", "dot"] = Field(
        default="cosine", alias="QDRANT_DISTANCE"
    )
    qdrant_timeout: float = Field(default=10.0, alias="QDRANT_TIMEOUT")
    qdrant_retry_count: int = Field(default=3, alias="QDRANT_RETRY_COUNT")
    qdrant_connection_pool_size: int = Field(
        default=32,
        alias="QDRANT_CONNECTION_POOL_SIZE",
        ge=1,
        le=64,  # Maximized for high-end hardware
    )
    # Unified batch configuration for optimal performance
    default_batch_size: int = Field(
        default=256,
        alias="DEFAULT_BATCH_SIZE",
        ge=64,
        le=512,
        description="Default batch size for all operations",
    )
    qdrant_batch_size: int = Field(
        default=256, alias="QDRANT_BATCH_SIZE", ge=64, le=512
    )
    qdrant_prefetch_size: int = Field(
        default=2048,
        alias="QDRANT_PREFETCH_SIZE",
        ge=256,
        le=4096,  # Doubled for better memory utilization
    )
    qdrant_search_exact: bool = Field(default=False, alias="QDRANT_SEARCH_EXACT")

    # Vector Service Configuration - using modular implementation

    # HF Text Embeddings Inference (TEI)
    tei_url: str = Field(default="http://localhost:8080", alias="TEI_URL")
    tei_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B", alias="TEI_MODEL")
    tei_max_concurrent_requests: int = Field(
        default=256,
        alias="TEI_MAX_CONCURRENT_REQUESTS",  # Doubled for RTX 4070 performance
    )
    tei_max_batch_tokens: int = Field(default=32768, alias="TEI_MAX_BATCH_TOKENS")
    tei_tokens_per_item: int = Field(
        default=30,
        alias="TEI_TOKENS_PER_ITEM",
        ge=10,
        le=100,
        description="Estimated tokens per embedding item for batch size calculation",
    )
    tei_batch_size: int = Field(
        default=512,
        alias="TEI_BATCH_SIZE",  # Increased for better GPU utilization
    )  # Will be validated against TEI_MAX_BATCH_TOKENS
    tei_timeout: float = Field(default=30.0, alias="TEI_TIMEOUT")

    # Embedding Configuration
    embedding_max_length: int = Field(default=32000, alias="EMBEDDING_MAX_LENGTH")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    embedding_max_retries: int = Field(default=3, alias="EMBEDDING_MAX_RETRIES")

    # Retry configuration with exponential backoff
    retry_initial_delay: float = Field(
        default=1.0,
        alias="RETRY_INITIAL_DELAY",
        ge=0.1,
        le=10.0,
        description="Initial delay in seconds for exponential backoff",
    )
    retry_max_delay: float = Field(
        default=60.0,
        alias="RETRY_MAX_DELAY",
        ge=1.0,
        le=300.0,
        description="Maximum delay in seconds for exponential backoff",
    )
    retry_exponential_base: float = Field(
        default=2.0,
        alias="RETRY_EXPONENTIAL_BASE",
        ge=1.1,
        le=5.0,
        description="Base for exponential backoff calculation",
    )
    embedding_workers: int = Field(
        default=8, alias="EMBEDDING_WORKERS", ge=1, le=16
    )  # Doubled for better concurrency

    # Chunking Configuration
    chunk_size: int = Field(default=1024, alias="CHUNK_SIZE", gt=0, le=32768)
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP", ge=0)
    word_to_token_ratio: float = Field(default=1.4, alias="WORD_TO_TOKEN_RATIO", gt=0)

    # Reranker Configuration
    reranker_model: str = Field(
        default="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", alias="RERANKER_MODEL"
    )
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_top_k: int = Field(default=10, alias="RERANKER_TOP_K", gt=0, le=100)
    reranker_max_length: int = Field(
        default=512, alias="RERANKER_MAX_LENGTH", gt=0, le=4096
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
        if not 1 <= v <= 16:
            raise ValueError("embedding_workers must be between 1 and 16")
        return v

    @field_validator("browser_pool_size")
    @classmethod
    def validate_browser_pool_size(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError("browser_pool_size must be between 1 and 20")
        return v

    @field_validator("browser_extra_args")
    @classmethod
    def validate_browser_extra_args(cls, v: list[str]) -> list[str]:
        # Whitelist of safe flags that won't break headless mode
        safe_flags = {
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu-sandbox",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features",
            "--process-per-site",
            "--aggressive-cache-discard",
            "--memory-pressure-off",
            "--no-zygote",
            "--max_old_space_size",
            "--js-flags",
            "--renderer-process-limit",
            "--max-renderer-processes",
            "--enable-accelerated-2d-canvas",
            "--enable-gpu-compositing",
            "--enable-gpu-rasterization",
            "--enable-zero-copy",
            "--enable-oop-rasterization",
        }

        for arg in v:
            if not arg.startswith("--"):
                raise ValueError(f"Browser arg must start with '--': {arg}")
            flag_name = arg.split("=")[0]
            if not any(flag_name.startswith(safe) for safe in safe_flags):
                logger = logging.getLogger(__name__)
                logger.warning(f"Potentially unsafe browser flag: {arg}")
        return v

    @field_validator("browser_hardware_profile")
    @classmethod
    def validate_browser_hardware_profile(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed_profiles = {"rtx4070_i7", "high_memory", "basic"}
        if v not in allowed_profiles:
            raise ValueError(
                f"browser_hardware_profile must be one of {allowed_profiles} or None"
            )
        return v

    # GPU flag validation removed - no longer using custom Chrome flags

    # Crawling Configuration
    crawl_headless: bool = Field(default=True, alias="CRAWL_HEADLESS")
    crawl_browser: str = Field(default="chromium", alias="CRAWL_BROWSER")
    crawl_max_pages: int = Field(default=1000, alias="CRAWL_MAX_PAGES")
    crawl_max_depth: int = Field(default=3, alias="CRAWL_MAX_DEPTH")
    max_concurrent_crawls: int = Field(
        default=25, alias="MAX_CONCURRENT_CRAWLS"
    )  # Restored to 25 for RTX 4070 + i7-13700k
    crawler_delay: float = Field(default=0.1, alias="CRAWLER_DELAY")
    crawler_timeout: float = Field(default=30.0, alias="CRAWLER_TIMEOUT")
    crawl_min_words: int = Field(default=50, alias="CRAWL_MIN_WORDS")
    crawl_remove_overlays: bool = Field(default=True, alias="CRAWL_REMOVE_OVERLAYS")
    crawl_extract_media: bool = Field(default=False, alias="CRAWL_EXTRACT_MEDIA")
    crawl_knowledge_graph: bool = Field(default=False, alias="CRAWL_KNOWLEDGE_GRAPH")
    crawl_user_agent: str = Field(
        default="Crawler-MCP/0.1.0 (+https://github.com/user/crawler-mcp)",
        alias="CRAWL_USER_AGENT",
    )

    # Resource blocking (performance)
    crawl_block_images: bool = Field(default=False, alias="CRAWL_BLOCK_IMAGES")
    crawl_block_media: bool = Field(default=False, alias="CRAWL_BLOCK_MEDIA")
    crawl_block_stylesheets: bool = Field(
        default=False, alias="CRAWL_BLOCK_STYLESHEETS"
    )
    crawl_block_fonts: bool = Field(default=False, alias="CRAWL_BLOCK_FONTS")

    # Crawl4AI 0.7.0 Advanced Features
    crawl_adaptive_mode: bool = Field(default=True, alias="CRAWL_ADAPTIVE_MODE")
    crawl_confidence_threshold: float = Field(
        default=0.8, alias="CRAWL_CONFIDENCE_THRESHOLD"
    )
    crawl_top_k_links: int = Field(default=5, alias="CRAWL_TOP_K_LINKS")
    crawl_url_seeding: bool = Field(default=True, alias="CRAWL_URL_SEEDING")
    crawl_score_threshold: float = Field(default=0.4, alias="CRAWL_SCORE_THRESHOLD")
    crawl_virtual_scroll: bool = Field(default=True, alias="CRAWL_VIRTUAL_SCROLL")
    crawl_scroll_count: int = Field(default=20, alias="CRAWL_SCROLL_COUNT")
    crawl_scroll_delay: int = Field(default=50, alias="CRAWL_SCROLL_DELAY")
    crawl_virtual_scroll_batch_size: int = Field(
        default=20,
        alias="CRAWL_VIRTUAL_SCROLL_BATCH_SIZE",  # Doubled for high-end hardware
    )
    crawl_memory_threshold: float = Field(
        default=70.0, alias="CRAWL_MEMORY_THRESHOLD"
    )  # Conservative threshold to prevent swapping

    # Performance Optimization Configuration
    crawl_enable_streaming: bool = Field(default=True, alias="CRAWL_ENABLE_STREAMING")
    crawl_enable_caching: bool = Field(
        default=True, alias="CRAWL_ENABLE_CACHING"
    )  # Re-enabled for performance with repeated URLs

    # Browser Configuration
    browser_extra_args: list[str] = Field(
        default_factory=list, alias="BROWSER_EXTRA_ARGS"
    )
    browser_hardware_profile: str | None = Field(
        default=None, alias="BROWSER_HARDWARE_PROFILE"
    )

    # High-Performance Configuration (i7-13700k + RTX 4070)
    browser_pool_size: int = Field(
        default=4, alias="BROWSER_POOL_SIZE", ge=1, le=20
    )  # Optimized for CPU efficiency (4 browsers x 6 threads = 24)
    file_processing_threads: int = Field(
        default=24,
        alias="FILE_PROCESSING_THREADS",
        ge=1,
        le=32,  # Increased for i7-13700k (24 threads)
    )
    crawl_concurrency: int = Field(
        default=8, alias="CRAWL_CONCURRENCY", ge=1, le=50
    )  # Optimized for CPU efficiency without oversubscription
    content_cache_size_gb: int = Field(
        default=16,
        alias="CONTENT_CACHE_SIZE_GB",
        ge=1,
        le=32,  # Doubled for 32GB+ RAM
    )

    # Alternative crawling approach settings
    use_arun_many_for_sitemaps: bool = Field(
        default=True,  # Enable by default for comprehensive sitemap crawling
        alias="USE_ARUN_MANY_FOR_SITEMAPS",
        description="Use arun_many() with sitemap URLs instead of BFSDeepCrawlStrategy",
    )
    max_concurrent_sessions: int = Field(
        default=15,  # Optimized for RTX 4070 + i7-13700k hardware
        alias="CRAWL_MAX_CONCURRENT_SESSIONS",
        ge=1,
        le=50,
        description="Maximum concurrent sessions for arun_many() approach",
    )

    # Crawl4AI Performance Optimizations
    crawl_text_mode: bool = Field(
        default=False,
        alias="CRAWL_TEXT_MODE",
        description="Enable text-only mode for 3-4x faster crawling (disables images)",
    )
    crawl_light_mode: bool = Field(
        default=True,
        alias="CRAWL_LIGHT_MODE",
        description="Enable light mode to optimize browser performance",
    )
    crawl_enable_gpu: bool = Field(
        default=True,
        alias="CRAWL_ENABLE_GPU",
        description="Enable GPU acceleration for browsers (optimized for performance)",
    )
    use_lxml_strategy: bool = Field(
        default=True,
        alias="USE_LXML_STRATEGY",
        description="Use LXMLWebScrapingStrategy for 20x faster parsing",
    )

    # URL Pattern Exclusions - Conservative defaults to avoid admin/auth areas and binary files
    crawl_exclude_url_patterns: list[str] = Field(
        default=[
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
            # Large binary file extensions
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
        ],
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
        default=0.25,
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
    deduplication_strategy: str = Field(
        default="content_hash",
        alias="DEDUPLICATION_STRATEGY",
        description="Deduplication strategy: 'content_hash', 'timestamp', or 'none'",
    )
    delete_orphaned_chunks: bool = Field(
        default=True,
        alias="DELETE_ORPHANED_CHUNKS",
        description="Delete chunks that no longer exist in re-crawled content",
    )
    preserve_unchanged_metadata: bool = Field(
        default=True,
        alias="PRESERVE_UNCHANGED_METADATA",
        description="Preserve metadata for unchanged chunks during re-crawls",
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
        default=10,
        alias="DIRECTORY_MAX_FILE_SIZE_MB",
        ge=1,
        le=100,
        description="Maximum file size in MB to process when crawling directories",
    )

    # CORS & Security
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    cors_credentials: bool = Field(default=True, alias="CORS_CREDENTIALS")
    max_request_size: int = Field(default=10485760, alias="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: float = Field(default=30.0, alias="REQUEST_TIMEOUT")

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
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.chunk_size > self.embedding_max_length:
            raise ValueError("CHUNK_SIZE must be <= EMBEDDING_MAX_LENGTH")
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
    def _validate_deduplication(self) -> "CrawlerMCPSettings":
        """Validate deduplication configuration."""
        valid_strategies = {"content_hash", "timestamp", "none"}
        if self.deduplication_strategy not in valid_strategies:
            raise ValueError(
                f"DEDUPLICATION_STRATEGY must be one of: {', '.join(valid_strategies)}"
            )

        # Warn if deduplication is disabled but orphan deletion is enabled
        if not self.deduplication_enabled and self.delete_orphaned_chunks:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "DELETE_ORPHANED_CHUNKS is enabled but DEDUPLICATION_ENABLED is False. "
                "Orphan detection requires deduplication to be enabled."
            )

        return self

    @model_validator(mode="after")
    def _populate_excluded_selectors(self) -> "CrawlerMCPSettings":
        """Populate crawl_excluded_selectors from property if it's empty."""
        if not self.crawl_excluded_selectors:
            self.crawl_excluded_selectors = self.crawl_excluded_selectors_list
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
