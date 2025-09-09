# RAG Pipeline Test Suite

This directory contains comprehensive testing tools for the Crawler-MCP RAG pipeline.

## Quick Start

### Using the Test Runner (Recommended)

```bash
# Run from project root
./run_rag_tests.sh                    # Full test suite (~10 min)
./run_rag_tests.sh -m quick          # Quick tests only (~2 min)
./run_rag_tests.sh -m stress -r report.json  # Stress tests with report
```

### Direct Python Execution

```bash
# From project root
python tests/test_full_rag_pipeline.py --mode full
python tests/test_full_rag_pipeline.py --mode quick --debug
python tests/test_full_rag_pipeline.py --mode stress --report-file results.json
```

## Test Modes

| Mode | Duration | Description |
|------|----------|-------------|
| `quick` | ~2 min | Basic functionality tests, health checks, minimal content |
| `full` | ~10 min | Comprehensive testing of all pipeline components |
| `stress` | ~20+ min | Performance benchmarks, concurrent queries, load testing |

## What Gets Tested

### 🏥 Service Health Checks
- TEI (Text Embeddings Inference) service connectivity
- Qdrant vector database availability
- RAG service initialization and model loading
- Reranker model availability (if enabled)

### 📥 Content Ingestion Pipeline
- Single page scraping with RAG ingestion
- Multi-page crawling and indexing
- Content chunking (tiktoken/character-based)
- Deduplication logic
- Metadata preservation

### 🗄️ Vector Storage Operations
- Embedding generation via TEI
- Vector storage in Qdrant collections
- Collection statistics and health
- Search functionality

### 🔍 RAG Query Processing
- Basic semantic search queries
- Filtered queries (by source, score threshold)
- Reranking with Qwen3 model
- Query result structure validation
- Performance metrics collection

### 🚀 Performance Benchmarks (Stress Mode)
- Concurrent query handling
- Latency measurements
- Throughput analysis
- Cache effectiveness
- Memory usage patterns

## Prerequisites

### Docker Services

Ensure the following services are running:

```bash
# Start required services
docker-compose up -d tei qdrant

# Verify services are healthy
curl http://localhost:8080/health  # TEI
curl http://localhost:6333/health  # Qdrant
```

### Environment Configuration

Required environment variables (set in `.env`):

```env
# TEI Configuration
TEI_URL=http://localhost:8080
TEI_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=crawler_mcp

# Optional: Reranking
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### Python Dependencies

All dependencies should be available in your virtual environment:

```bash
uv sync  # Install all dependencies
source .venv/bin/activate
```

## Test Output

### Console Output

The test suite provides colored, real-time progress output:

```
🕷️  RAG Pipeline Test Suite
Mode: FULL

============================================================
                    SERVICE HEALTH CHECKS
============================================================
  [ PASS  ] TEI Embeddings Service
  [ PASS  ] Qdrant Vector Database
  [ PASS  ] RAG Service Initialization

============================================================
                   CONTENT INGESTION TESTS
============================================================
  [ PASS  ] Single Page Scraping + RAG
  [ PASS  ] Multi-page Crawling
```

### Performance Metrics

Key performance indicators are tracked and reported:

- **Ingestion Speed**: Pages processed per second
- **Embedding Latency**: Time to generate embeddings
- **Query Response Time**: End-to-end query processing
- **Cache Hit Rate**: Query cache effectiveness
- **Concurrent Performance**: Multi-query handling

### Test Reports

Generate detailed JSON reports for analysis:

```bash
./run_rag_tests.sh -r test_report.json
```

Report structure:
```json
{
  "test_summary": {
    "mode": "full",
    "total_tests": 15,
    "passed_tests": 14,
    "success_rate": 93.3,
    "duration": 623.45
  },
  "performance_metrics": {
    "scrape_time": 2.34,
    "total_chunks": 45,
    "embedding_dimension": 384,
    "query_0_performance": {
      "total_time": 0.156,
      "matches": 5,
      "avg_score": 0.73
    }
  },
  "errors": [],
  "configuration": {
    "tei_url": "http://localhost:8080",
    "qdrant_url": "http://localhost:6333"
  }
}
```

## Troubleshooting

### Common Issues

**Services Not Running**
```bash
# Check service status
docker-compose ps

# Start missing services
docker-compose up -d tei qdrant

# View service logs
docker-compose logs tei
docker-compose logs qdrant
```

**Network Connectivity Issues**
```bash
# Test direct service connectivity
curl -v http://localhost:8080/health
curl -v http://localhost:6333/health

# Check firewall/proxy settings
```

**Environment Variable Issues**
```bash
# Verify environment loading
python -c "from crawler_mcp.config import settings; print(settings.tei_url)"

# Check .env file exists and is readable
ls -la .env
```

**Memory/Performance Issues**
```bash
# Monitor resource usage during tests
docker stats

# Run with reduced concurrency
./run_rag_tests.sh -m quick  # Use quick mode for limited resources
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
./run_rag_tests.sh -d  # Enable debug logging
```

This provides detailed information about:
- Service initialization steps
- HTTP requests/responses
- Embedding generation process
- Vector storage operations
- Query processing pipeline

### Manual Testing

For specific component testing, you can also run individual pytest tests:

```bash
# Run existing pytest suite
pytest tests/test_rag_tools.py -v

# Run specific test functions
pytest tests/test_services.py::test_embedding_service -v
```

## Integration with CI/CD

The test suite is designed to work in automated environments:

```yaml
# Example GitHub Actions step
- name: Run RAG Pipeline Tests
  run: |
    docker-compose up -d tei qdrant
    sleep 30  # Wait for services to be ready
    ./run_rag_tests.sh -m quick -r ci_report.json
  env:
    TEI_URL: http://localhost:8080
    QDRANT_URL: http://localhost:6333
```

## Contributing

When adding new test cases:

1. Add test methods to the appropriate section in `test_full_rag_pipeline.py`
2. Update performance metrics collection if needed
3. Ensure cleanup code handles new test resources
4. Update this documentation with any new prerequisites or usage patterns

The test suite follows these patterns:
- Use async/await for all service interactions
- Record performance metrics for analysis
- Provide clear pass/fail criteria
- Clean up all test data after completion
- Handle service failures gracefully
