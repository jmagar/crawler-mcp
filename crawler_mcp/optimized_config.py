"""
Configuration for the optimized high-performance web crawler.

Centralized settings for URL discovery, concurrency, content extraction, and
performance parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Literal

# Valid monitor mode values
VALID_MONITOR_MODES = ["DETAILED", "AGGREGATED"]


@dataclass
class OptimizedConfig:
    """Configuration for optimized crawler with high-performance defaults"""

    # URL Discovery Configuration
    max_urls_to_discover: int = 1000
    """Maximum number of URLs to discover from sitemaps before crawling"""

    url_score_threshold: float = 0.3
    """Minimum relevance score for URLs to be included in crawling"""

    discover_from_sitemap: bool = True
    """Whether to discover URLs from sitemap.xml files"""

    discover_from_robots: bool = True
    """Whether to parse robots.txt for additional sitemap locations"""

    # Enhanced Sitemap Configuration
    sitemap_follow_redirects: bool = True
    """Follow HTTP redirects when fetching sitemaps"""

    sitemap_accept_compressed: bool = True
    """Accept and decompress gzipped sitemaps (.xml.gz files)"""

    sitemap_max_redirects: int = 5
    """Maximum number of redirects to follow for sitemaps"""

    sitemap_timeout_seconds: int = 15
    """Timeout for individual sitemap fetch operations"""

    sitemap_user_agent: str = "Mozilla/5.0 (compatible; OptimizedCrawler/1.0)"
    """User agent string for sitemap requests"""

    # Concurrency Configuration
    max_concurrent_crawls: int = 16
    """Maximum number of concurrent crawling sessions (optimal for i7-13700K)"""

    memory_threshold: float = 70.0
    """Memory usage threshold percentage before reducing concurrency"""

    use_aggressive_mode: bool = False
    """Enable aggressive mode with higher concurrency and memory usage"""

    # Content Extraction Configuration
    content_threshold: float = 0.48
    """Content relevance threshold for PruningContentFilter"""

    min_word_count: int = 50
    """Minimum word count for valid page content"""

    excluded_tags: list[str] = field(
        default_factory=lambda: [
            "nav",
            "header",
            "footer",
            "aside",
            "script",
            "style",
            "noscript",
            "iframe",
        ]
    )
    """HTML tags to exclude from content extraction"""

    # Performance Configuration
    enable_cache: bool = True
    """Enable crawl4ai caching for improved performance"""

    browser_headless: bool = True
    """Run browser in headless mode for better performance"""

    text_mode: bool = True
    """Disable image loading for maximum crawling speed"""

    browser_mode: Literal["full", "text", "minimal"] = "full"
    """
    Browser mode for crawling:
    - 'full': Standard browser with all features (JS enabled)
    - 'text': Text-only mode (lightweight, no JS/images)
    - 'minimal': Most aggressive resource blocking
    """

    # Behavior Configuration (User Requested)
    check_robots_txt: bool = False
    """Ignore robots.txt rules - aggressive crawling as requested"""

    use_rate_limiting: bool = False
    """Disable rate limiting for maximum speed - as requested"""

    respect_crawl_delay: bool = False
    """Ignore crawl delay directives for maximum throughput"""

    # Timeout Configuration
    page_timeout: int = 30000
    """Page load timeout in milliseconds"""

    discovery_timeout: int = 10
    """Timeout for sitemap/robots.txt discovery in seconds"""

    # Advanced Performance Options
    browser_pool_size: int = 4
    """Number of browser instances in pool (for non-builtin mode)"""

    virtual_scroll_enabled: bool = False
    """Enable virtual scrolling for dynamic content (reduces performance)"""

    stealth_mode: bool = False
    """Enable anti-detection features (may reduce performance)"""

    # Content Processing Options
    extract_links: bool = True
    """Extract links from crawled pages"""

    extract_images: bool = False
    """Extract image URLs from crawled pages"""

    extract_metadata: bool = True
    """Extract page metadata (title, description, etc.)"""

    # Content Filtering
    enable_content_filter: bool = False
    """Enable content filtering for cleaner markdown generation"""

    content_filter_type: Literal["pruning", "bm25", "none"] = "pruning"
    """Type of content filter to use"""

    pruning_threshold: float = 0.45
    """Pruning filter threshold (0-1, higher = more aggressive filtering)"""

    pruning_threshold_type: Literal["fixed", "dynamic"] = "dynamic"
    """Pruning threshold type - dynamic adjusts based on content"""

    pruning_min_words: int = 10
    """Minimum words required for a content block to be kept"""

    bm25_threshold: float = 1.2
    """BM25 relevance threshold for query-based filtering"""

    bm25_user_query: str = ""
    """User query for BM25 content filtering (empty = auto-detect from page)"""

    # Embeddings (HF TEI)
    enable_embeddings: bool = False
    """Generate embeddings for page content via TEI"""

    tei_endpoint: str = "http://localhost:8080"
    """Base URL for TEI server (e.g., http://steamy-wsl:8080)"""

    tei_model_name: str = "Qwen3-0.6B"
    """Logical model name to annotate in metadata."""

    tei_batch_size: int = 16
    """Batch size for TEI requests"""

    tei_timeout_s: float = 15.0
    """Timeout seconds for TEI requests"""

    tei_max_retries: int = 1
    """Retries for TEI calls"""

    tei_parallel_requests: int = 4
    """Max concurrent TEI batch requests (post-crawl embedding phase)"""

    # TEI input shaping (for GPU utilization)
    tei_max_input_chars: int = 0
    """Max characters per input text sent to TEI (truncate client-side). 0 = no truncation (default)."""

    tei_target_chars_per_batch: int = 64000
    """Target total characters per TEI batch to better saturate GPU"""

    tei_collapse_whitespace: bool = True
    """Collapse excessive whitespace before embedding to reduce tokens"""

    # TEI server capacity hints (align with compose if provided)
    tei_max_batch_tokens: int = 0
    """Server max batch tokens; if > 0, pack batches by token estimate"""

    tei_max_client_batch_size: int = 128
    """Upper bound of items per client request (server --max-client-batch-size)"""

    tei_max_concurrent_requests: int = 0
    """Server max concurrent requests; if > 0, cap client parallel to this"""

    tei_chars_per_token: float = 4.0
    """Approx chars per token for token-budget estimation (language dependent)"""

    # Reranking (TEI-compatible reranker)
    enable_rerank: bool = False
    """Enable reranking of retrieval results"""

    rerank_endpoint: str = ""
    """Reranker base URL (defaults to tei_endpoint if empty)"""

    rerank_model_name: str = ""
    """Reranker model identifier (optional)"""

    rerank_topk: int = 5
    """Number of items to keep after reranking"""

    # Embedding post-processing
    embedding_target_dim: int = 0
    """If > 0, project embeddings to this dimension (truncate/pad)"""

    embedding_projection: str = "none"
    """Projection method: 'none' | 'truncate' | 'pad_zero'"""

    # Qdrant (Vector store)
    enable_qdrant: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "crawler_pages"
    qdrant_distance: str = "Cosine"
    qdrant_vectors_name: str = ""
    qdrant_batch_size: int = 128
    qdrant_parallel_requests: int = 2
    qdrant_upsert_wait: bool = True
    qdrant_api_key: str | None = None
    qdrant_vector_size: int = 0  # 0 = infer from embeddings

    # Quality Control
    hash_placeholder_detection: bool = True
    """Enable detection and filtering of hash placeholder content"""

    content_validation: bool = True
    """Enable content quality validation"""

    # Language filtering
    allowed_locales: list[str] = field(default_factory=lambda: ["en"])
    """Restrict crawling to these locale prefixes (e.g., ['en']). Empty = no filter.

    A URL like '/fr/docs/...' has locale 'fr'. '/zh-CN/...' has 'zh-CN'. If a locale
    segment is absent (e.g., '/docs/intro'), it matches '' and also matches 'en' by
    default when 'en' is in the allowed list (treating no-locale as default English).
    """

    # Validation Relaxation (Docs)
    doc_relax_validation_patterns: list[str] = field(
        default_factory=lambda: [r"/python-sdk/"]
    )
    """List of regex patterns; if a page URL matches any and content would be
    rejected for doc-like reasons, accept it anyway (e.g., '/python-sdk/')."""

    # Follow internal links (BFS) after first batch
    follow_internal_links: bool = True
    """After initial batch, schedule internal links from returned pages (bounded)."""

    follow_internal_budget: int = 200
    """Max number of internal links to enqueue in the follow-up pass."""

    # Output Management Configuration
    output_dir: str = "./output"
    """Base output directory for all crawl results and reports"""

    crawler_monitor_mode: Literal["DETAILED", "AGGREGATED"] = "AGGREGATED"
    """Monitoring detail level: DETAILED for per-page logs, AGGREGATED for summary stats"""

    max_domain_backups: int = 1
    """Number of backup copies to keep per domain (latest + N backups)"""

    max_output_size_gb: float = 1.0
    """Maximum total size of output directory in GB before cleanup"""

    log_rotation_size_mb: int = 10
    """Log file rotation size in MB"""

    log_rotation_backups: int = 3
    """Number of rotated log backups to keep (.1, .2, .3)"""

    cache_retention_hours: int = 24
    """Cache file retention time in hours"""

    auto_cleanup: bool = True
    """Enable automatic cleanup when size limits are exceeded"""

    # Advanced Performance Configuration
    enable_light_mode: bool = True
    """Enable light_mode in browser for background feature disabling"""

    viewport_width: int = 1280
    """Browser viewport width for optimal rendering"""

    viewport_height: int = 720
    """Browser viewport height for optimal rendering"""

    enable_javascript: bool = True
    """Enable JavaScript execution (disable for text-only crawling)"""

    browser_extra_args: list[str] = field(
        default_factory=lambda: [
            "--disable-dev-shm-usage",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=TranslateUI",
        ]
    )
    """Extra browser arguments for performance optimization"""

    # CrawlerRunConfig Performance Options
    cache_strategy: Literal["enabled", "bypass", "disabled", "adaptive"] = "disabled"
    """Caching strategy: enabled=always cache, bypass=never read cache, disabled=no cache, adaptive=smart caching"""

    wait_condition: Literal["domcontentloaded", "networkidle"] = "domcontentloaded"
    """Page load completion condition for faster crawling"""

    html_delay_seconds: float = 0.05
    """Delay before capturing HTML content (reduced from default 0.1s)"""

    enable_text_only_mode: bool = False
    """Extract text-only content for maximum speed"""

    enable_streaming_mode: bool = True
    """Enable streaming mode for real-time processing and reduced memory usage"""

    excluded_selectors: list[str] = field(
        default_factory=lambda: [
            "#ads",
            ".advertisement",
            ".tracking",
            ".social-share",
            ".cookie-banner",
            ".newsletter-signup",
            "[class*='ad-']",
            "[id*='ad-']",
            ".sidebar-ad",
            ".sponsored",
        ]
    )
    """CSS selectors for elements to exclude (ads, tracking, etc.)"""

    exclude_external_links: bool = True
    """Remove external links from crawled content"""

    remove_forms: bool = False
    """Remove form elements when they're not needed"""

    exclude_external_images: bool = True
    """Exclude images from external domains"""

    crawl_semaphore_count: int = 25
    """Semaphore count for controlling crawl concurrency (increased for better throughput)"""

    mean_request_delay: float = 0.02
    """Mean delay between requests in seconds (reduced for faster crawling)"""

    max_request_delay_range: float = 0.05
    """Maximum random delay range for request pacing (reduced for faster crawling)"""

    # Memory Management Configuration
    memory_threshold_percent: float = 60.0
    """Memory usage threshold percentage before reducing concurrency"""

    check_interval: float = 0.5
    """Interval in seconds for checking memory and session status"""

    max_session_permit: int = 20
    """Maximum number of concurrent crawler sessions allowed"""

    memory_wait_timeout: float = 30.0
    """Timeout in seconds when waiting for memory to free up"""

    # Content-Type Specific Configurations
    enable_url_based_optimization: bool = True
    """Enable URL pattern-based configuration optimization"""

    documentation_patterns: list[str] = field(
        default_factory=lambda: [
            "*/docs/*",
            "*/documentation/*",
            "*/api/*",
            "*/guide/*",
            "*/tutorial/*",
            "*/reference/*",
            "*/manual/*",
        ]
    )
    """URL patterns that indicate documentation content"""

    # All compatibility flags removed - using only documented Crawl4AI APIs

    @classmethod
    def from_env(cls, prefix: str = "OPTIMIZED_CRAWLER_") -> "OptimizedConfig":
        """
        Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix

        Returns:
            OptimizedConfig instance with values from environment
        """
        config = cls()

        # Map of attribute names to environment variable suffixes
        env_mappings = {
            "max_urls_to_discover": "MAX_URLS",
            "max_concurrent_crawls": "MAX_CONCURRENT",
            "memory_threshold": "MEMORY_THRESHOLD",
            "content_threshold": "CONTENT_THRESHOLD",
            "min_word_count": "MIN_WORDS",
            "page_timeout": "PAGE_TIMEOUT",
            "discovery_timeout": "DISCOVERY_TIMEOUT",
            "browser_headless": "HEADLESS",
            "text_mode": "TEXT_MODE",
            "browser_mode": "BROWSER_MODE",
            "enable_cache": "ENABLE_CACHE",
            "use_aggressive_mode": "AGGRESSIVE_MODE",
            "stealth_mode": "STEALTH_MODE",
            # Embeddings
            "enable_embeddings": "EMBEDDINGS",
            "tei_endpoint": "TEI_ENDPOINT",
            "tei_model_name": "TEI_MODEL_NAME",
            "tei_batch_size": "TEI_BATCH_SIZE",
            "tei_timeout_s": "TEI_TIMEOUT_S",
            "tei_max_retries": "TEI_MAX_RETRIES",
            "tei_parallel_requests": "TEI_PARALLEL_REQUESTS",
            "tei_max_input_chars": "TEI_MAX_INPUT_CHARS",
            "tei_target_chars_per_batch": "TEI_TARGET_CHARS_PER_BATCH",
            "tei_collapse_whitespace": "TEI_COLLAPSE_WHITESPACE",
            "tei_max_batch_tokens": "TEI_MAX_BATCH_TOKENS",
            "tei_max_client_batch_size": "TEI_MAX_CLIENT_BATCH_SIZE",
            "tei_max_concurrent_requests": "TEI_MAX_CONCURRENT_REQUESTS",
            "tei_chars_per_token": "TEI_CHARS_PER_TOKEN",
            # Rerank
            "enable_rerank": "RERANK",
            "rerank_endpoint": "RERANK_ENDPOINT",
            "rerank_model_name": "RERANK_MODEL",
            "rerank_topk": "RERANK_TOPK",
            "embedding_target_dim": "EMBEDDING_TARGET_DIM",
            "embedding_projection": "EMBEDDING_PROJECTION",
            # Qdrant
            "enable_qdrant": "QDRANT",
            "qdrant_url": "QDRANT_URL",
            "qdrant_collection": "QDRANT_COLLECTION",
            "qdrant_distance": "QDRANT_DISTANCE",
            "qdrant_vectors_name": "QDRANT_VECTORS_NAME",
            "qdrant_batch_size": "QDRANT_BATCH_SIZE",
            "qdrant_parallel_requests": "QDRANT_PARALLEL_REQUESTS",
            "qdrant_upsert_wait": "QDRANT_UPSERT_WAIT",
            "qdrant_api_key": "QDRANT_API_KEY",
            "qdrant_vector_size": "QDRANT_VECTOR_SIZE",
            # Enhanced sitemap options
            "sitemap_follow_redirects": "SITEMAP_FOLLOW_REDIRECTS",
            "sitemap_accept_compressed": "SITEMAP_ACCEPT_COMPRESSED",
            "sitemap_max_redirects": "SITEMAP_MAX_REDIRECTS",
            "sitemap_timeout_seconds": "SITEMAP_TIMEOUT_SECONDS",
            "sitemap_user_agent": "SITEMAP_USER_AGENT",
            # Output Management
            "output_dir": "OUTPUT_DIR",
            "max_domain_backups": "MAX_DOMAIN_BACKUPS",
            "max_output_size_gb": "MAX_OUTPUT_SIZE_GB",
            "log_rotation_size_mb": "LOG_ROTATION_SIZE_MB",
            "log_rotation_backups": "LOG_ROTATION_BACKUPS",
            "cache_retention_hours": "CACHE_RETENTION_HOURS",
            "auto_cleanup": "AUTO_CLEANUP",
            # Monitor & strategy preferences
            "enable_crawler_monitor": "ENABLE_CRAWLER_MONITOR",
            "crawler_monitor_mode": "CRAWLER_MONITOR_MODE",
            "crawler_monitor_max_visible_rows": "CRAWLER_MONITOR_MAX_VISIBLE_ROWS",
            "use_http_strategy_when_no_js": "USE_HTTP_STRATEGY_WHEN_NO_JS",
            # Performance options
            "enable_light_mode": "ENABLE_LIGHT_MODE",
            "viewport_width": "VIEWPORT_WIDTH",
            "viewport_height": "VIEWPORT_HEIGHT",
            "enable_javascript": "ENABLE_JAVASCRIPT",
            "cache_strategy": "CACHE_STRATEGY",
            "wait_condition": "WAIT_CONDITION",
            "html_delay_seconds": "HTML_DELAY_SECONDS",
            "enable_text_only_mode": "ENABLE_TEXT_ONLY_MODE",
            "exclude_external_links": "EXCLUDE_EXTERNAL_LINKS",
            "remove_forms": "REMOVE_FORMS",
            "exclude_external_images": "EXCLUDE_EXTERNAL_IMAGES",
            "crawl_semaphore_count": "CRAWL_SEMAPHORE_COUNT",
            "mean_request_delay": "MEAN_REQUEST_DELAY",
            "max_request_delay_range": "MAX_REQUEST_DELAY_RANGE",
            "enable_url_based_optimization": "ENABLE_URL_BASED_OPTIMIZATION",
            # Memory management options
            "memory_threshold_percent": "MEMORY_THRESHOLD_PERCENT",
            "check_interval": "CHECK_INTERVAL",
            "max_session_permit": "MAX_SESSION_PERMIT",
            "memory_wait_timeout": "MEMORY_WAIT_TIMEOUT",
        }

        # Load configuration from environment
        for attr_name, env_suffix in env_mappings.items():
            env_var = f"{prefix}{env_suffix}"
            env_value = os.getenv(env_var)

            if env_value is not None:
                # Get the current attribute value to determine type
                current_value = getattr(config, attr_name)

                # Convert based on type
                if isinstance(current_value, bool):
                    setattr(
                        config,
                        attr_name,
                        env_value.lower() in ("true", "1", "yes", "on"),
                    )
                elif isinstance(current_value, int):
                    setattr(config, attr_name, int(env_value))
                elif isinstance(current_value, float):
                    setattr(config, attr_name, float(env_value))
                else:
                    # Special handling for crawler_monitor_mode to normalize case
                    if attr_name == "crawler_monitor_mode":
                        normalized_value = env_value.upper()
                        valid_modes = VALID_MONITOR_MODES
                        if normalized_value not in valid_modes:
                            raise ValueError(
                                f"Invalid {env_var}: {env_value!r}. "
                                f"Must be one of {valid_modes} (case-insensitive)."
                            )
                        setattr(config, attr_name, normalized_value)
                    else:
                        setattr(config, attr_name, env_value)

        # Always ensure these values as requested
        config.check_robots_txt = False
        config.use_rate_limiting = False
        config.respect_crawl_delay = False

        # Validate configuration and fail fast if invalid
        validation_errors = config.validate()
        if validation_errors:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {error}" for error in validation_errors)
            )

        return config

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of configuration
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result

    def update_from_dict(self, data: dict[str, Any]) -> None:
        """
        Update configuration from dictionary.

        Args:
            data: Dictionary with configuration updates
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_aggressive_config(self) -> "OptimizedConfig":
        """
        Get configuration optimized for maximum performance.

        Returns:
            New OptimizedConfig instance with aggressive settings
        """
        aggressive = OptimizedConfig()

        # Copy current settings
        for key, value in self.__dict__.items():
            setattr(aggressive, key, value)

        # Override with aggressive settings
        aggressive.max_concurrent_crawls = 28
        aggressive.memory_threshold = 85.0
        aggressive.use_aggressive_mode = True
        aggressive.page_timeout = 15000  # Shorter timeout
        aggressive.discovery_timeout = 5  # Faster discovery
        aggressive.browser_pool_size = 8
        # Aggressive performance settings
        aggressive.enable_light_mode = True
        aggressive.enable_javascript = False  # Disable JS for speed
        aggressive.cache_strategy = "enabled"
        aggressive.wait_condition = "domcontentloaded"
        aggressive.html_delay_seconds = 0.01
        aggressive.enable_text_only_mode = True
        aggressive.exclude_external_links = True
        aggressive.remove_forms = True
        aggressive.exclude_external_images = True
        aggressive.crawl_semaphore_count = 40
        aggressive.mean_request_delay = 0.01
        aggressive.max_request_delay_range = 0.02
        # Aggressive memory settings
        aggressive.memory_threshold_percent = 80.0
        aggressive.check_interval = 0.3
        aggressive.max_session_permit = 30
        aggressive.memory_wait_timeout = 20.0

        return aggressive

    def get_conservative_config(self) -> "OptimizedConfig":
        """
        Get configuration optimized for reliability over speed.

        Returns:
            New OptimizedConfig instance with conservative settings
        """
        conservative = OptimizedConfig()

        # Copy current settings
        for key, value in self.__dict__.items():
            setattr(conservative, key, value)

        # Override with conservative settings
        conservative.max_concurrent_crawls = 8
        conservative.memory_threshold = 60.0
        conservative.use_aggressive_mode = False
        conservative.page_timeout = 60000  # Longer timeout
        conservative.discovery_timeout = 15
        conservative.content_validation = True
        conservative.hash_placeholder_detection = True
        # Conservative performance settings
        conservative.enable_light_mode = False
        conservative.enable_javascript = True  # Keep JS enabled
        conservative.cache_strategy = "enabled"
        conservative.wait_condition = "networkidle"
        conservative.html_delay_seconds = 0.2
        conservative.enable_text_only_mode = False
        conservative.exclude_external_links = False
        conservative.remove_forms = False
        conservative.exclude_external_images = False
        conservative.crawl_semaphore_count = 10
        conservative.mean_request_delay = 0.1
        conservative.max_request_delay_range = 0.3
        # Conservative memory settings
        conservative.memory_threshold_percent = 50.0
        conservative.check_interval = 1.0
        conservative.max_session_permit = 10
        conservative.memory_wait_timeout = 60.0

        return conservative

    def validate(self) -> list[str]:
        """
        Validate configuration values and return list of issues.

        Returns:
            List of validation error messages
        """
        errors = []

        if self.max_concurrent_crawls < 1:
            errors.append("max_concurrent_crawls must be at least 1")

        if self.memory_threshold < 10.0 or self.memory_threshold > 95.0:
            errors.append("memory_threshold must be between 10.0 and 95.0")

        if self.content_threshold < 0.0 or self.content_threshold > 1.0:
            errors.append("content_threshold must be between 0.0 and 1.0")

        if self.min_word_count < 0:
            errors.append("min_word_count must be non-negative")

        if self.page_timeout < 1000:
            errors.append("page_timeout must be at least 1000ms")

        if self.discovery_timeout < 1:
            errors.append("discovery_timeout must be at least 1 second")

        if not self.excluded_tags:
            errors.append("excluded_tags should not be empty for optimal performance")

        # Validate crawler_monitor_mode
        valid_monitor_modes = VALID_MONITOR_MODES
        if self.crawler_monitor_mode not in valid_monitor_modes:
            errors.append(
                f"crawler_monitor_mode must be one of {valid_monitor_modes}, "
                f"got: {self.crawler_monitor_mode!r}"
            )

        return errors

    def __str__(self) -> str:
        """String representation of configuration"""
        return (
            f"OptimizedConfig(concurrent={self.max_concurrent_crawls}, "
            f"memory={self.memory_threshold}%, "
            f"robots_txt={self.check_robots_txt}, "
            f"rate_limit={self.use_rate_limiting})"
        )
