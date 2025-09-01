# Optimized Crawler - High-Performance Web Crawling System

A sophisticated, production-ready web crawling system built on Crawl4AI with advanced features including GPU-aware TEI embeddings, Qdrant vector storage, adaptive concurrency control, and comprehensive content extraction capabilities.

## Overview

This crawler represents a complete overhaul of traditional web scraping approaches, providing:

- **High-Performance Parallel Crawling**: Adaptive concurrency tuning based on system metrics
- **Intelligent Content Extraction**: Multiple extraction strategies with hash placeholder prevention
- **GPU-Optimized Embeddings**: TEI (Text Embeddings Inference) integration with intelligent batching
- **Vector Storage Integration**: Seamless Qdrant upsert with verification and semantic search
- **Production-Ready Monitoring**: Comprehensive metrics, progress tracking, and health monitoring
- **Language-Aware Processing**: Locale filtering with configurable language support
- **Extensible Architecture**: Clean separation of concerns with factory patterns and strategy interfaces

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
crawler_mcp/crawlers/optimized/
├── run.py                      # CLI entry point with comprehensive argument parsing
├── config.py                   # Central configuration with 100+ options
├── AGENTS.md                   # Agent interaction specifications
├── .env.example                # Environment configuration template
│
├── core/                       # Core orchestration and execution engine
│   ├── strategy.py            # Main crawler strategy (AsyncCrawlerStrategy)
│   ├── parallel_engine.py     # High-performance parallel execution
│   └── adaptive_dispatcher.py # Dynamic concurrency tuning
│
├── factories/                  # Component factory patterns
│   ├── browser_factory.py     # Browser configuration factory
│   ├── content_extractor.py   # Content extraction configurations
│   └── dispatcher_factory.py  # Crawler dispatcher creation
│
├── processing/                 # Content and URL processing
│   ├── url_discovery.py       # Sitemap parsing and URL discovery
│   └── result_converter.py    # Result format conversion
│
├── clients/                    # External service integrations
│   ├── tei_client.py          # TEI embeddings client
│   ├── qdrant_http_client.py  # Qdrant vector database client
│   └── local_reranker.py      # Local reranking capabilities
│
└── utils/                      # Utilities and monitoring
    └── monitoring.py           # System metrics and health monitoring
```

## Installation & Setup

### Requirements
- Python 3.11+
- uv package manager (recommended) or pip
- Running TEI server for embeddings (optional but recommended)
- Qdrant vector database (optional)

### Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Configure your environment variables:
```bash
# Core crawler settings
OPTIMIZED_CRAWLER_MAX_CONCURRENT=28
OPTIMIZED_CRAWLER_JAVASCRIPT_ENABLED=true
OPTIMIZED_CRAWLER_CHECK_ROBOTS_TXT=false

# TEI Embeddings settings
OPTIMIZED_CRAWLER_EMBEDDINGS=true
OPTIMIZED_CRAWLER_TEI_ENDPOINT=http://localhost:8080
OPTIMIZED_CRAWLER_TEI_MAX_CONCURRENT=80
OPTIMIZED_CRAWLER_TEI_MAX_CLIENT_BATCH=128
OPTIMIZED_CRAWLER_TEI_MAX_BATCH_TOKENS=163840

# Qdrant Vector Storage
OPTIMIZED_CRAWLER_QDRANT=true
OPTIMIZED_CRAWLER_QDRANT_URL=http://localhost:6333
OPTIMIZED_CRAWLER_QDRANT_COLLECTION=web_content

# Language filtering (default English-only)
OPTIMIZED_CRAWLER_ALLOWED_LOCALES=en
```

### Quick Start

Basic crawl with progress monitoring:
```bash
uv run python -m crawler_mcp.crawlers.optimized.run \
  --url https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-overview \
  --progress
```

## Configuration Reference

The system supports over 100 configuration options organized into logical groups:

### URL Discovery & Site Exploration
```bash
--max-crawl-depth 3              # Maximum crawl depth
--max-urls-per-sitemap 10000     # Sitemap URL limit
--max-fallback-urls 100          # Fallback link discovery limit
--min-content-chars 100          # Minimum content length threshold
--max-content-chars 100000       # Maximum content length limit
```

### Concurrency & Performance
```bash
--max-concurrent 28              # Maximum concurrent crawls
--batch-size 50                  # Processing batch size
--page-timeout 30000             # Page timeout in milliseconds
--enable-adaptive-concurrency    # Enable dynamic concurrency tuning
--cpu-threshold 0.92             # CPU threshold for concurrency reduction
--error-rate-threshold 0.20      # Error rate threshold for tuning
```

### Content Extraction
```bash
--javascript true                # Enable JavaScript rendering
--extract-images false           # Include images in content
--text-mode false               # Text-only extraction mode
--min-word-count 50             # Minimum word count threshold
--excluded-tags script,style,nav # HTML tags to exclude
```

### Language & Locale Filtering
```bash
--allowed-locales en             # Allowed locale codes
--allowed-locales all            # Allow all languages
--locale-detection-threshold 0.8 # Locale detection confidence
```

### TEI Embeddings Configuration
```bash
--embeddings true                # Enable embeddings generation
--tei-endpoint http://localhost:8080
--tei-model sentence-transformers/all-MiniLM-L6-v2
--tei-max-concurrent 80          # TEI parallel requests
--tei-max-client-batch 128       # Client-side batching
--tei-max-batch-tokens 163840    # Token budget per batch
--tei-max-input-chars 8000       # Input truncation limit
--tei-collapse-ws true           # Collapse whitespace
--tei-timeout 15.0               # Request timeout
```

### Qdrant Vector Storage
```bash
--qdrant true                    # Enable Qdrant integration
--qdrant-url http://localhost:6333
--qdrant-collection web_content
--qdrant-vector-size 384         # Vector dimensions
--qdrant-batch-size 100          # Upsert batch size
--qdrant-parallel-requests 4     # Parallel upsert requests
--verify-qdrant                  # Verify uploads after crawl
```

### Monitoring & Logging
```bash
--progress                       # Live progress updates
--per-page-log                   # Log each page result
--verbose                        # Verbose logging output
--heartbeat-interval 2.0         # Progress update frequency
--enable-cache                   # Enable crawl cache
```

## Core Components

### Strategy Pattern (core/strategy.py)
The main orchestrator that inherits from Crawl4AI's `AsyncCrawlerStrategy`:
- Coordinates all crawler components
- Manages the complete crawl pipeline
- Handles error recovery and retries
- Integrates with monitoring systems

### Parallel Execution Engine (core/parallel_engine.py)
High-performance crawling engine featuring:
- Utilizes Crawl4AI's `arun_many()` for optimal throughput
- Comprehensive metrics collection via `CrawlStats` dataclass
- Memory-efficient batch processing
- Error isolation and recovery

### Adaptive Concurrency Control (core/adaptive_dispatcher.py)
Dynamic performance tuning system:
- **ConcurrencyTuner**: Monitors system metrics and adjusts parallelism
- **CPU Monitoring**: Reduces concurrency when CPU > 92%
- **Error Rate Control**: Scales back on error rates > 20%
- **Automatic Recovery**: Gradually increases concurrency as conditions improve

### Content Extraction Factory (factories/content_extractor.py)
Sophisticated content extraction with multiple strategies:
- **Default Generator**: Balanced extraction with hash placeholder prevention
- **Aggressive Generator**: Maximum content extraction for challenging pages
- **Minimal Generator**: Speed-optimized extraction for basic content
- **Content-Type Specific**: Optimized configurations for articles, documentation, products, news, and blogs

### URL Discovery System (processing/url_discovery.py)
Intelligent URL discovery with multiple strategies:
- **Primary**: Comprehensive sitemap parsing with gzip support, namespace handling, and redirect following
- **Secondary**: JavaScript-powered link harvesting from page content
- **Filtering**: Locale-aware URL filtering and deduplication
- **Validation**: Robots.txt compliance and URL validation

### TEI Integration (clients/tei_client.py)
Production-ready embeddings client:
- **Dual Protocol Support**: Native TEI `/embed` and OpenAI-compatible `/v1/embeddings`
- **Intelligent Batching**: Token-aware batching with configurable limits
- **Error Handling**: Automatic retry logic with exponential backoff
- **Connection Management**: Efficient aiohttp session handling

### Qdrant Integration (clients/qdrant_http_client.py)
High-performance vector database client:
- **Adaptive Batching**: Dynamic batch sizing based on server capacity
- **Verification System**: Post-upload verification with detailed reporting
- **Named/Unnamed Vectors**: Support for different vector configurations
- **Health Monitoring**: Connection health checks and error recovery

## Advanced Features

### Intelligent Content Filtering
The system prevents hash placeholders and ensures high-quality content through:
- Dynamic content filtering with configurable thresholds
- Preservation of document structure while removing noise
- Intelligent tag exclusion based on content type
- Word count and content length validation

### Language-Aware Processing
Comprehensive locale support featuring:
- Path-based locale detection (e.g., `/en/docs`, `/fr/articles`)
- Configurable allowed locales with wildcard support
- Default English-only filtering with override capability
- Locale confidence scoring and validation

### GPU-Optimized Embeddings
TEI integration optimized for GPU workloads:
- Token-budget aware batching to maximize GPU utilization
- Parallel request processing with server capacity awareness
- Input preprocessing including whitespace collapse and truncation
- Automatic fallback between TEI and OpenAI protocols

### Vector Storage Optimization
Qdrant integration designed for production scale:
- Adaptive batch sizing to fully utilize server `parallel_requests` capacity
- Vector dimension inference from embeddings or manual configuration
- Comprehensive verification system with detailed mismatch reporting
- Support for both named and unnamed vector configurations

### Real-Time Monitoring
Production-grade monitoring and observability:
- Live progress updates with throughput calculations
- System resource monitoring (CPU, memory, process metrics)
- Per-page logging with success/failure tracking
- Comprehensive final reports with performance analytics

## Performance Optimization

### Concurrency Tuning
- **Adaptive Scaling**: Automatic concurrency adjustment based on system load
- **CPU Awareness**: Reduces load when system CPU > 92%
- **Error Recovery**: Backs off on high error rates, gradually recovers
- **Resource Optimization**: Balances throughput with system stability

### Memory Management
- **Streaming Processing**: Processes results in batches to manage memory
- **Connection Pooling**: Efficient HTTP connection reuse
- **Cache Utilization**: Leverages Crawl4AI caching for repeat crawls
- **Garbage Collection**: Proper cleanup of resources and connections

### TEI Optimization
- **Batch Sizing**: Intelligent batching based on token budgets and item counts
- **Parallel Processing**: Concurrent requests within server limits
- **Input Preprocessing**: Whitespace collapse and truncation for efficiency
- **Connection Management**: Persistent connections with proper lifecycle

### Qdrant Optimization
- **Adaptive Batching**: Batch sizes that fully utilize server parallel capacity
- **Verification Strategies**: Configurable post-upload verification
- **Connection Efficiency**: HTTP/2 support and connection reuse
- **Error Handling**: Robust retry logic with exponential backoff

## Integration Guides

### TEI Server Setup
```bash
# Start TEI server with GPU support
docker run --gpus all -p 8080:80 \
  --volume $PWD/data:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id sentence-transformers/all-MiniLM-L6-v2
```

### Qdrant Database Setup
```bash
# Start Qdrant with persistence
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant:latest
```

### Environment Integration
The crawler integrates seamlessly with existing environments through:
- Environment variable configuration with `OPTIMIZED_CRAWLER_` prefix
- CLI flag override capability for dynamic configuration
- Support for `.env` and `.env.local` files
- Configuration validation and error reporting

## CLI Reference

### Basic Usage
```bash
uv run python -m crawler_mcp.crawlers.optimized.run [OPTIONS]
```

### Primary Commands
- `--url URL`: Target URL to crawl (required unless using search)
- `--qdrant-search QUERY`: Semantic search mode (no crawling)
- `--verify-qdrant`: Verify Qdrant uploads after crawling
- `--progress`: Enable real-time progress monitoring
- `--per-page-log`: Log individual page results
- `--verbose`: Enable verbose logging output

### Configuration Overrides
- `--max-concurrent N`: Override concurrent crawl limit
- `--javascript true/false`: Enable/disable JavaScript rendering
- `--embeddings true/false`: Enable/disable embedding generation
- `--qdrant true/false`: Enable/disable Qdrant integration
- `--allowed-locales CODES`: Set allowed language locales

### Output Control
- `--output-file PATH`: Save results to specific file
- `--output-format json/ndjson`: Control output format
- `--include-metadata`: Include crawl metadata in output

## Developer Guide

### Extending the System
The modular architecture supports easy extension:

1. **Custom Content Extractors**: Extend `ContentExtractorFactory` with new extraction strategies
2. **Additional Clients**: Implement new service clients following the existing patterns
3. **Custom Strategies**: Create new crawler strategies by inheriting from the base strategy
4. **Monitoring Extensions**: Add new metrics to the monitoring system
5. **Processing Pipelines**: Extend the result conversion and processing pipelines

### Code Quality Standards
- **Linting**: `uv run ruff check .`
- **Formatting**: `uv run ruff format .`
- **Type Checking**: `uv run mypy crawler_mcp`
- **Testing**: `uv run pytest -m "not slow and not requires_services"`

### Architecture Principles
- **Single Responsibility**: Each component has a clear, focused purpose
- **Dependency Injection**: Configuration and dependencies are injected rather than hardcoded
- **Error Isolation**: Failures in one component don't cascade to others
- **Observable**: Comprehensive logging and metrics throughout the system
- **Configurable**: All behavior is configurable through environment or CLI

## Troubleshooting

### Common Issues

**Sitemap Discovery Issues**
- Ensure `--verbose` flag is used to see sitemap extraction messages
- Check robots.txt compliance if `--check-robots-txt` is enabled
- Verify sitemap URLs are accessible and properly formatted

**JavaScript Rendering Problems**
- Enable JavaScript with `--javascript true` for dynamic content
- Increase `--page-timeout` for slow-loading pages
- Check browser factory configuration for proper setup

**TEI Connection Issues**
- Verify TEI server is running and accessible at configured endpoint
- Check TEI server logs for capacity or model loading issues
- Adjust `--tei-max-concurrent` based on server capabilities

**Qdrant Integration Problems**
- Confirm Qdrant is running and collection exists
- Verify vector dimensions match between TEI model and Qdrant collection
- Use `--verify-qdrant` to check upload success

**Performance Issues**
- Monitor system resources and adjust `--max-concurrent` accordingly
- Enable `--enable-adaptive-concurrency` for automatic tuning
- Consider `--text-mode true` for faster extraction
- Use `--min-word-count` to filter low-quality content

### Debug Modes
- `--verbose`: Enable detailed logging
- `--per-page-log`: Log each page processing result
- `--progress`: Monitor real-time crawling progress
- `--verify-qdrant`: Verify all vector uploads

## API Reference

### Configuration Classes
- `OptimizedConfig`: Central configuration with 100+ options
- `CrawlStats`: Comprehensive crawling statistics
- `ConcurrencyTuner`: Adaptive concurrency control

### Core Interfaces
- `AsyncCrawlerStrategy`: Main crawler strategy interface
- `ContentExtractorFactory`: Content extraction factory
- `UrlDiscoveryService`: URL discovery and processing
- `TEIEmbeddingsClient`: TEI embeddings client
- `QdrantHTTPClient`: Qdrant vector database client

### Key Methods
- `run_crawl()`: Execute complete crawl pipeline
- `discover_urls()`: Intelligent URL discovery
- `embed_texts()`: Generate text embeddings
- `upsert_vectors()`: Store vectors in Qdrant
- `search_vectors()`: Semantic search functionality

---

This crawler represents a production-ready solution for large-scale web crawling with advanced features for modern AI and search applications. The modular architecture ensures maintainability while the comprehensive configuration system provides flexibility for diverse use cases.

For questions, feature requests, or contributions, please refer to the project documentation or open an issue in the repository.
