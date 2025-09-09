# RAG Pipeline Test Suite - Enhanced Crawl Tool Integration

## Overview

The test suite has been updated to use the **exact same interface and behavior** as your actual `crawl` and `scrape` tools, ensuring perfect alignment between testing and production crawling.

## Enhanced Features

### 🕷️ Exact Crawl Tool Interface

The tests now use the precise parameter signatures from your crawl tools:

#### Scrape Tool Parameters
```python
await client.call_tool("scrape", {
    "url": "https://httpbin.org/html",
    "screenshot": False,
    "wait_for": None,
    "css_selector": None,
    "javascript": False,
    "timeout_ms": 30000,
    "rag_ingest": True,
})
```

#### Crawl Tool Parameters
```python
await client.call_tool("crawl", {
    "target": "https://httpbin.org/html",  # Can be website, repo, directory
    "limit": 5,
    "depth": 2,
    "max_concurrent": 3,
    "include_patterns": None,
    "exclude_patterns": None,
    "javascript": False,
    "screenshot_samples": 0,
    "timeout_ms": 60000,
    "rag_ingest": True,
})
```

### 🎯 Multi-Target Testing

The suite now tests all supported target types:

1. **Website Crawling** - HTTP/HTTPS URLs with depth and concurrency
2. **Directory Crawling** - Local filesystem scanning with pattern matching
3. **GitHub Repository Crawling** - Clone and index Git repositories
4. **GitHub PR Crawling** - Target specific pull requests

### 📊 Enhanced Metrics Collection

The test suite now captures detailed metrics from actual crawl operations:

#### Scrape Metrics
- Response time and success rate
- Word count and content quality
- RAG ingestion status

#### Crawl Metrics
- Pages processed vs failed
- Crawl duration and throughput
- Target type detection accuracy
- Configuration effectiveness

#### Performance Comparisons
- Basic vs concurrent crawling
- JavaScript enabled vs disabled
- Different timeout configurations

### 🔧 Target Detection Validation

Tests the crawl tool's intelligent target detection:

```python
# Website detection
"https://httpbin.org/html" → kind: "website"

# Repository detection
"https://github.com/octocat/Hello-World.git" → kind: "repository"

# Directory detection
"/path/to/local/dir" → kind: "directory"

# GitHub PR detection
"https://github.com/octocat/Hello-World/pull/1" → kind: "github_pr"
```

### 📁 File Pattern Matching

Directory and repository crawling tests include realistic patterns:

```python
"include_patterns": ["**/*.md", "**/*.py", "**/*.js"],
"exclude_patterns": ["**/node_modules/**", "**/.git/**", "**/__pycache__/**"]
```

## Test Modes & Coverage

### Quick Mode (~2 min)
- Basic scraping functionality
- Simple website crawling
- Core RAG queries
- Service health checks

### Full Mode (~10 min)
- All quick mode tests
- Multi-target crawling (website + directory)
- Filtered RAG queries
- Target detection validation
- Parameter validation

### Stress Mode (~20+ min)
- All full mode tests
- GitHub repository crawling
- Concurrent query performance
- Crawl configuration comparison
- Directory crawling with file creation

## Real-World Test Scenarios

### 1. Production Crawl Simulation
```bash
# Test exactly like production usage
./run_rag_tests.sh -m full -r production_test.json
```

### 2. Performance Benchmarking
```bash
# Stress test with detailed metrics
./run_rag_tests.sh -m stress -r benchmark_results.json
```

### 3. Quick Validation
```bash
# Fast smoke test before deployment
./run_rag_tests.sh -m quick
```

## Enhanced Reporting

### Performance Metrics
The test report now includes crawl-specific metrics:

```json
{
  "performance_metrics": {
    "scrape_time": 2.34,
    "crawl_time": 12.67,
    "crawl_pages_processed": 5,
    "crawl_pages_failed": 0,
    "crawl_duration": 8.45,
    "crawl_docs_found": 5,
    "dir_files_processed": 3,
    "repo_files_processed": 15,
    "crawl_performance_comparison": {
      "basic": {"time": 8.2, "success": true, "pages": 3},
      "concurrent": {"time": 5.1, "success": true, "pages": 3},
      "javascript": {"time": 12.3, "success": true, "pages": 2}
    }
  }
}
```

### Success Validation
Tests verify the exact response structure from your crawl tools:

- `success` boolean status
- `kind` target type detection
- `stats.processed` page count
- `stats.failed` failure count
- `stats.duration_s` timing
- `docs_preview` content samples
- `rag.enabled` ingestion status

## Usage Examples

### Test Website Crawling
```python
# Tests your exact crawl tool with website target
result = await client.call_tool("crawl", {
    "target": "https://example.com",
    "limit": 10,
    "depth": 2,
    "max_concurrent": 3,
    "javascript": False,
    "rag_ingest": True
})
```

### Test Directory Crawling
```python
# Tests local filesystem crawling
result = await client.call_tool("crawl", {
    "target": "/path/to/docs",
    "limit": 50,
    "include_patterns": ["**/*.md", "**/*.rst"],
    "exclude_patterns": ["**/build/**"],
    "rag_ingest": True
})
```

### Test Repository Crawling
```python
# Tests Git repository indexing
result = await client.call_tool("crawl", {
    "target": "https://github.com/user/repo.git",
    "limit": 100,
    "include_patterns": ["**/*.py", "**/*.md"],
    "exclude_patterns": ["**/.git/**", "**/venv/**"],
    "rag_ingest": True
})
```

## Benefits

### ✅ Perfect Production Alignment
- Uses identical tool interfaces
- Tests actual crawl behavior
- Validates real response formats

### ✅ Comprehensive Coverage
- All target types supported
- Parameter validation
- Error handling scenarios

### ✅ Performance Insights
- Comparative benchmarking
- Configuration optimization
- Bottleneck identification

### ✅ Reliable Testing
- Real service integration
- No mocking or simulation
- Production-grade validation

## Next Steps

The test suite now perfectly mirrors your production crawl tools, giving you confidence that:

1. **Crawl behavior is consistent** between test and production
2. **All target types work correctly** (website, directory, repository, PR)
3. **Parameter validation is robust** across different configurations
4. **Performance characteristics are measurable** and optimizable
5. **RAG integration works end-to-end** with actual content ingestion

Run the tests to validate your entire RAG pipeline with production-identical crawling behavior!
