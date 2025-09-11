"""Unified crawler configuration - consolidates all actually used settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from crawler_mcp.constants import (
    # Batch Processing
    DEFAULT_CHUNK_OVERLAP,
    # Content Processing
    DEFAULT_CHUNK_SIZE,
    # Concurrency
    DEFAULT_CONNECTION_POOL_SIZE,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MAX_LENGTH,
    DEFAULT_EMBEDDING_WORKERS,
    DEFAULT_EXCLUDED_SELECTORS,
    # Excluded patterns
    DEFAULT_EXCLUDED_TAGS,
    DEFAULT_EXCLUDED_URL_PATTERNS,
    # Server & Network
    DEFAULT_HOST,
    DEFAULT_MAX_CONCURRENT_CRAWLS,
    # Crawling
    DEFAULT_MAX_CRAWL_PAGES,
    DEFAULT_MIN_WORD_COUNT,
    DEFAULT_PAGE_TIMEOUT,
    DEFAULT_PORT,
    DEFAULT_PRUNING_THRESHOLD,
    DEFAULT_REQUEST_TIMEOUT,
    # Retry
    DEFAULT_RETRY_COUNT,
    DEFAULT_TIMEOUT,
    DEFAULT_VIEWPORT_HEIGHT,
    # Browser
    DEFAULT_VIEWPORT_WIDTH,
    MAX_CHUNK_SIZE,
    MAX_REQUEST_SIZE_BYTES,
    MIN_CHUNK_SIZE,
    QDRANT_DEFAULT_TIMEOUT,
    RETRY_EXPONENTIAL_BASE,
    RETRY_INITIAL_DELAY,
    RETRY_MAX_DELAY,
    TEI_DEFAULT_BATCH_SIZE,
    TEI_DEFAULT_TIMEOUT,
    TEI_MAX_BATCH_TOKENS,
    TEI_MAX_CONCURRENT_REQUESTS,
    # Enums
    BrowserMode,
    CacheStrategy,
    ContentFilterType,
    ExtractionStrategy,
    LogLevel,
    ProxyType,
    WaitCondition,
)


class CrawlerSettings(BaseSettings):
    """Unified crawler configuration - all actually used settings."""

    # Server/Transport Settings (10 settings)
    transport: str = "http"
    server_host: str = DEFAULT_HOST  # Was "0.0.0.0"
    server_port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    server_name: str = "crawler-mcp"
    version: str = "0.1.0"
    allow_origins: list[str] = ["*"]
    server_metrics_enabled: bool = True
    request_timeout: int = Field(default=DEFAULT_REQUEST_TIMEOUT, ge=1)
    log_to_file: bool = False
    uvicorn_access_log: bool = True

    # Qdrant Settings (9 settings)
    qdrant_url: str = "http://localhost:7000"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "crawled_pages"
    qdrant_timeout: int = Field(default=QDRANT_DEFAULT_TIMEOUT, ge=1)
    qdrant_search_exact: bool = False
    qdrant_vector_size: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSION,
        description="Must match embedding_dimension",
    )
    qdrant_connection_pool_size: int = Field(
        default=DEFAULT_CONNECTION_POOL_SIZE, ge=1, le=64
    )
    qdrant_vectors_name: str | None = None
    qdrant_upsert_wait: bool = True

    # TEI Settings (7 settings)
    tei_url: str = Field(default="http://localhost:8080", alias="TEI_URL")
    tei_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="TEI_MODEL")
    tei_max_retries: int = Field(
        default=DEFAULT_RETRY_COUNT, ge=0, alias="TEI_MAX_RETRIES"
    )
    tei_retry_delay: float = Field(
        default=RETRY_INITIAL_DELAY, ge=0.1, alias="TEI_RETRY_DELAY"
    )
    tei_timeout: int = Field(default=TEI_DEFAULT_TIMEOUT, ge=1, alias="TEI_TIMEOUT")
    tei_batch_size: int = Field(
        default=TEI_DEFAULT_BATCH_SIZE, ge=1, alias="TEI_BATCH_SIZE"
    )
    tei_max_concurrent_requests: int = Field(
        default=TEI_MAX_CONCURRENT_REQUESTS, ge=1, alias="TEI_MAX_CONCURRENT_REQUESTS"
    )
    tei_max_batch_tokens: int = Field(
        default=TEI_MAX_BATCH_TOKENS, ge=1, alias="TEI_MAX_BATCH_TOKENS"
    )

    # Embedding Settings (7 settings)
    embedding_dimension: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSION, description="Must be 384, 768, or 1024"
    )
    embedding_max_length: int = Field(default=DEFAULT_EMBEDDING_MAX_LENGTH, ge=1)
    embedding_normalize: bool = True
    embedding_max_retries: int = Field(default=DEFAULT_RETRY_COUNT, ge=0)
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE, ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE
    )
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)
    embedding_workers: int = Field(default=DEFAULT_EMBEDDING_WORKERS, ge=1, le=16)
    enable_embeddings: bool = True  # Enable by default if TEI is configured
    enable_qdrant: bool = True  # Enable by default if Qdrant is configured

    # Crawler Settings (37 settings)
    max_depth: int = Field(default=3, ge=1, le=10)
    max_pages: int = Field(default=DEFAULT_MAX_CRAWL_PAGES, ge=1, alias="MAX_PAGES")
    page_timeout: int = Field(
        default=int(DEFAULT_PAGE_TIMEOUT / 1000), ge=1
    )  # Convert ms to seconds
    delay_between_requests: float = Field(default=1.0, ge=0.0)
    concurrent_requests: int = Field(default=3, ge=1, le=50)
    max_retries: int = Field(default=DEFAULT_RETRY_COUNT, ge=0)
    retry_delay: float = Field(default=RETRY_INITIAL_DELAY, ge=0.1)
    max_content_length: int = Field(default=MAX_REQUEST_SIZE_BYTES, ge=1024)
    user_agent: str = "CrawlerBot/1.0"
    # DEPRECATION: prefer respect_robots_txt; ignore_robots_txt is derived
    ignore_robots_txt: bool = True
    follow_redirects: bool = True
    allowed_domains: list[str] = []
    excluded_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_URL_PATTERNS)
    )
    include_external_links: bool = False
    min_content_length: int = Field(default=DEFAULT_MIN_WORD_COUNT, ge=1)
    crawl_exclude_url_patterns: list[str] = Field(default_factory=list)
    enable_streaming: bool = True  # Re-enabled now that cache is set to BYPASS
    extract_links: bool = True
    extract_images: bool = False

    # Content Filtering Settings (20 additional settings)
    excluded_selectors: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_SELECTORS)
    )
    excluded_tags: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_TAGS)
    )
    enable_content_filter: bool = True
    content_filter_type: str = ContentFilterType.PRUNING
    wait_condition: str = WaitCondition.DOM_CONTENT_LOADED
    html_delay_seconds: float = 2.0
    enable_text_only_mode: bool = False
    exclude_external_links: bool = False
    remove_forms: bool = False
    exclude_external_images: bool = False
    crawl_semaphore_count: int = 5
    mean_request_delay: float = 1.0
    max_request_delay_range: float = 2.0
    pruning_threshold: float = Field(default=DEFAULT_PRUNING_THRESHOLD, ge=0.0, le=1.0)
    pruning_threshold_type: str = "fixed"
    pruning_min_words: int = Field(default=DEFAULT_MIN_WORD_COUNT, ge=1)
    bm25_user_query: str | None = None
    bm25_threshold: float = 0.5
    enable_url_based_optimization: bool = False

    # Deduplication Settings (2 settings)
    deduplication_enabled: bool = True
    delete_orphaned_chunks: bool = False

    # Reranker Settings (3 settings)
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_fallback_to_custom: bool = True

    # Legacy Cache Settings (will be replaced by Enhanced Cache Settings below)
    cache_enabled: bool = True
    cache_backend: str = "memory"

    # Error Handling (6 settings)
    error_threshold: int = 10
    backoff_factor: float = Field(default=RETRY_EXPONENTIAL_BASE, ge=1.1, le=5.0)
    max_backoff: int = Field(default=int(RETRY_MAX_DELAY), ge=1)
    retry_initial_delay: float = Field(default=RETRY_INITIAL_DELAY, ge=0.1, le=10.0)
    retry_max_delay: float = Field(default=RETRY_MAX_DELAY, ge=1.0, le=300.0)
    retry_exponential_base: float = Field(
        default=RETRY_EXPONENTIAL_BASE, ge=1.1, le=5.0
    )

    # OAuth Settings (6 settings)
    oauth_enabled: bool = False
    oauth_provider: str | None = None
    google_base_url: str = "http://localhost:8000"
    google_scopes_list: list[str] = ["openid", "email", "profile"]

    # Logging & Debug (4 settings)
    log_level: str = "INFO"
    debug: bool = False
    log_file: str | None = None
    uvicorn_log_level: str = "info"

    # Server Resource Settings (1 setting)
    max_concurrent_crawls: int = Field(
        default=DEFAULT_MAX_CONCURRENT_CRAWLS, ge=1, le=50
    )

    # Browser Settings (12 new settings)
    browser_type: str = "chromium"
    browser_mode: BrowserMode = (
        BrowserMode.HEADLESS
    )  # Default to headless for WSL2 compatibility
    browser_width: int = Field(default=DEFAULT_VIEWPORT_WIDTH, ge=800)
    browser_height: int = Field(default=DEFAULT_VIEWPORT_HEIGHT, ge=600)
    browser_user_agent: str | None = None
    browser_timeout: float = Field(default=float(DEFAULT_PAGE_TIMEOUT / 1000), ge=1.0)
    browser_wait_for: float = Field(default=0.5, ge=0.0)
    browser_sleep_on_close: float = Field(default=0.5, ge=0.0)
    browser_js_enabled: bool = True
    browser_accept_downloads: bool = False
    browser_downloads_path: str | None = None
    browser_ignore_https_errors: bool = True

    # Crawler Orchestration Settings (13 new settings)
    crawl_delay: float = Field(default=1.0, ge=0.0)
    respect_robots_txt: bool = False
    max_redirects: int = Field(default=5, ge=1, le=20)
    exclude_patterns: list[str] = Field(default_factory=list)
    include_patterns: list[str] = Field(default_factory=list)
    css_selector: str | None = None
    word_threshold: int = Field(default=DEFAULT_MIN_WORD_COUNT, ge=1)
    only_text: bool = False
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.BASIC
    session_id: str | None = None
    override_navigator: bool = True

    # Enhanced Cache Settings (7 new settings, replaces existing 4)
    cache_strategy: CacheStrategy = (
        CacheStrategy.BYPASS  # BYPASS to avoid content being stored as hash references
    )  # Changed from ENABLED to fix content hash issue
    cache_ttl: int = 3600
    cache_dir: Path = Path(".cache")
    redis_url: str | None = None
    redis_db: int = 0
    redis_prefix: str = "crawler"
    cache_max_size: int = 1000

    # Proxy Settings (6 new settings)
    proxy_enabled: bool = False
    proxy_url: str | None = None
    proxy_type: ProxyType = ProxyType.HTTP
    proxy_username: str | None = None
    proxy_password: str | None = None
    proxy_rotation: bool = False

    # Performance Settings (8 new settings)
    connect_timeout: float = Field(default=10.0, ge=1.0)
    read_timeout: float = Field(default=DEFAULT_TIMEOUT, ge=1.0)
    retry_attempts: int = Field(
        default=DEFAULT_RETRY_COUNT, ge=0
    )  # Alias for max_retries
    memory_limit_mb: int = Field(default=1024, ge=256)
    disk_limit_mb: int = Field(default=5120, ge=1024)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables not in the model
        populate_by_name=True,  # Allow both field name and alias for env vars
    )

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: int) -> int:
        if v not in [384, 768, 1024]:
            raise ValueError("Embedding dimension must be 384, 768, or 1024")
        return v

    @field_validator("qdrant_url", "tei_url")
    @classmethod
    def validate_urls(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            return f"http://{v}"
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        # Use LogLevel enum values for validation
        valid_levels = [level.value for level in LogLevel]
        v_lower = v.lower()
        if v_lower not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_lower

    @field_validator("uvicorn_log_level")
    @classmethod
    def validate_uvicorn_log_level(cls, v: str) -> str:
        valid_levels = ["critical", "error", "warning", "info", "debug", "trace"]
        v_lower = v.lower()
        if v_lower not in valid_levels:
            raise ValueError(f"uvicorn_log_level must be one of {valid_levels}")
        return v_lower

    # Note: _ensure_positive validator removed - Field() constraints now handle validation

    @model_validator(mode="after")
    def _sync_robots_flags(self) -> CrawlerSettings:
        # Use pydantic v2's model_fields_set to detect fields explicitly provided.
        provided = getattr(self, "model_fields_set", set())
        provided_ignore = "ignore_robots_txt" in provided
        provided_respect = "respect_robots_txt" in provided

        # If both were explicitly set and are not logical opposites, fail fast.
        if provided_ignore and provided_respect:
            if self.ignore_robots_txt == (not self.respect_robots_txt):
                return self
            raise ValueError(
                "ignore_robots_txt conflicts with respect_robots_txt; set only one (prefer respect_robots_txt)."
            )

        # Derive the unset flag (prefer respect_robots_txt as canonical).
        if provided_respect:
            object.__setattr__(self, "ignore_robots_txt", not self.respect_robots_txt)
        elif provided_ignore:
            object.__setattr__(self, "respect_robots_txt", not self.ignore_robots_txt)
        else:
            object.__setattr__(self, "ignore_robots_txt", not self.respect_robots_txt)
        return self

    @model_validator(mode="after")
    def _validate_vector_dims(self) -> CrawlerSettings:
        """Ensure qdrant_vector_size matches embedding_dimension."""
        if self.qdrant_vector_size != self.embedding_dimension:
            raise ValueError(
                "qdrant_vector_size must equal embedding_dimension "
                f"(got {self.qdrant_vector_size} vs {self.embedding_dimension})."
            )
        return self


# Singleton with lazy loading to prevent circular imports
_settings: CrawlerSettings | None = None


def get_settings() -> CrawlerSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = CrawlerSettings()
    return _settings


# Prefer calling get_settings() to avoid eager init at import time.
# If strict backwards compatibility is required, keep this; otherwise remove.
# settings = get_settings()


def __getattr__(name: str):
    """Support dynamic attribute access for smooth imports."""
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
