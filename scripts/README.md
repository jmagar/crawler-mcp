# Qdrant Test Collection Cleanup Scripts

This directory contains scripts for managing and cleaning up test collections in Qdrant.

## Problem Solved

The test suite was accumulating test collections in Qdrant without proper cleanup, leading to 265+ orphaned collections. This was caused by:

1. Test fixtures creating collections with inconsistent naming patterns
2. Cleanup not happening when tests failed or were interrupted
3. No session-level cleanup to catch orphaned collections

## Solution Implemented

### 1. **Immediate Cleanup Script** (`cleanup_test_collections.py`)

**Purpose**: One-time cleanup of all existing test collections.

**Usage**:
```bash
# Interactive mode (asks for confirmation)
python scripts/cleanup_test_collections.py

# Force mode (skip confirmation, good for CI)
python scripts/cleanup_test_collections.py --force

# Show help
python scripts/cleanup_test_collections.py --help
```

**Features**:
- Identifies test collections by pattern matching
- Protects production collections (`crawlerr_documents`, `crawler_pages`, `documents_test`)
- Provides detailed logging of operations
- Safe to run multiple times

### 2. **Maintenance Script** (`maintenance_cleanup.py`)

**Purpose**: Periodic cleanup that can be run via cron job or manually.

**Usage**:
```bash
# Run maintenance cleanup
python scripts/maintenance_cleanup.py

# Show help
python scripts/maintenance_cleanup.py --help
```

**Features**:
- Gentle on Qdrant (adds delays between operations)
- Comprehensive logging for monitoring
- Safe to run periodically
- Only deletes test collections, never production ones

### 3. **Improved Test Fixtures** (`tests/conftest.py`)

**Changes Made**:
- **Consistent Naming**: Test collections now use `test_crawler_{uuid}` pattern
- **Global Tracking**: All created collections are tracked in a session-wide set
- **Multiple Cleanup Layers**:
  - Primary cleanup after each test
  - Finalizer cleanup if test fails
  - Session-level cleanup for any missed collections
  - Pre-test cleanup to remove orphaned collections from previous runs

**Benefits**:
- Collections are cleaned up even if tests crash
- Session cleanup catches anything missed
- Pre-test cleanup prevents accumulation between test runs

## Test Collection Patterns

The scripts recognize these patterns as test collections:

- `test_crawler_*` - Main test collections
- `test_github_pr_*` - GitHub PR related tests
- `test_aggressive_*`, `test_conservative_*` - Strategy tests
- `test_distance_cosine_*` - Distance metric tests
- `batch_test_*`, `scale_test_*`, `reconnect_test_*` - Performance tests
- `multipage_test_*`, `recovery_test_*` - Feature tests
- `concurrent_test_*`, `rag_test_*` - Concurrent/RAG tests
- `github_test_*`, `e2e_test_*` - Integration tests
- `filtered_rag_*`, `docs_test_*`, `perf_test_*` - Specialized tests
- `test_collection_*` - Legacy pattern from old fixtures

## Protected Collections

These collections are **never deleted** by any script:

- `crawlerr_documents` - Main production collection
- `crawler_pages` - Production page collection
- `documents_test` - Intentional test collection

## Verification

To check the current state of Qdrant collections:

```bash
curl -s "http://localhost:7000/collections" | jq '.result.collections'
```

## Results

Before cleanup: **265 collections** (262 test + 3 production)
After cleanup: **3 collections** (0 test + 3 production)

The cleanup was 100% successful, removing all 262 orphaned test collections while preserving all production collections.

## Automation Suggestions

### Cron Job (Optional)
Add to crontab for periodic cleanup:
```bash
# Run maintenance cleanup every hour
0 * * * * cd /path/to/crawl-mcp && python scripts/maintenance_cleanup.py >> logs/maintenance.log 2>&1

# Or daily at 2 AM
0 2 * * * cd /path/to/crawl-mcp && python scripts/maintenance_cleanup.py >> logs/maintenance.log 2>&1
```

### Pre-commit Hook (Optional)
Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: cleanup-test-collections
      name: Clean up test collections
      entry: python scripts/maintenance_cleanup.py
      language: system
      pass_filenames: false
      stages: [pre-push]
```

## Monitoring

Both scripts provide detailed logging. Key indicators:

- ✅ Successful operations
- ⚠️ Warnings (non-critical issues)
- ❌ Errors (operations that failed)

Monitor logs for patterns that might indicate new sources of collection leakage.
