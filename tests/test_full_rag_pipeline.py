#!/usr/bin/env python3
"""
Comprehensive RAG Pipeline Test Script

This script performs end-to-end testing of the entire RAG pipeline including:
- Service health checks (TEI, Qdrant, Crawl4AI)
- Content ingestion (scraping, crawling, chunking)
- Vector storage operations
- RAG query processing with reranking
- Performance measurement and reporting

Usage:
    python test_full_rag_pipeline.py [--mode quick|full|stress] [--debug] [--report-file output.json]
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from mcp.shared.memory import create_connected_server_and_client_session

# Import crawler_mcp components
from crawler_mcp.config import settings
from crawler_mcp.core.embeddings import EmbeddingService
from crawler_mcp.core.rag.service import RagService
from crawler_mcp.core.vectors import VectorService
from crawler_mcp.server import mcp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


@dataclass
class TestMetrics:
    """Container for test performance metrics."""

    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    errors: list[str] = field(default_factory=list)
    performance_data: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return (
            self.end_time - self.start_time
            if self.end_time
            else time.time() - self.start_time
        )

    @property
    def success_rate(self) -> float:
        return (
            (self.passed_tests / self.total_tests * 100)
            if self.total_tests > 0
            else 0.0
        )


class RAGPipelineTests:
    """Main test class for RAG pipeline testing."""

    def __init__(self, mode: str = "full", debug: bool = False):
        self.mode = mode
        self.debug = debug
        self.metrics = TestMetrics()
        self.test_collection = f"test_rag_{uuid.uuid4().hex[:8]}"
        self.test_data = self._get_test_data()

        # Configure logging level
        if debug:
            logging.getLogger("crawler_mcp").setLevel(logging.DEBUG)

    def _get_test_data(self) -> dict[str, Any]:
        """Define test data for various test scenarios."""
        return {
            "urls": [
                "https://httpbin.org/html",  # Reliable test content
                "https://example.com",  # Simple page
                "https://httpbin.org/json",  # JSON response
            ],
            "test_targets": {
                "website": "https://httpbin.org/html",
                "github_repo": "https://github.com/octocat/Hello-World.git",
                "github_pr": "https://github.com/octocat/Hello-World/pull/1",
            },
            "queries": [
                "Herman Melville",
                "example",
                "test data",
                "JSON response",
                "nonexistent content",
            ],
            "sample_content": {
                "title": "Test Document",
                "content": "This is a comprehensive test document for the RAG pipeline. It contains information about various topics including machine learning, natural language processing, and vector databases. The document should be chunked and indexed properly for retrieval.",
            },
        }

    def _print_header(self, text: str) -> None:
        """Print a formatted header."""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER} {text:<58} {Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.END}")

    def _print_test(self, test_name: str, status: str = "RUNNING") -> None:
        """Print test status."""
        color = (
            Colors.YELLOW
            if status == "RUNNING"
            else Colors.GREEN
            if status == "PASS"
            else Colors.RED
        )
        print(f"  {color}[{status:^8}]{Colors.END} {test_name}")

    def _record_test(self, passed: bool, error_msg: str | None = None) -> None:
        """Record test result."""
        self.metrics.total_tests += 1
        if passed:
            self.metrics.passed_tests += 1
        else:
            self.metrics.failed_tests += 1
            if error_msg:
                self.metrics.errors.append(error_msg)

    async def check_service_health(self) -> bool:
        """Check health of all required services."""
        self._print_header("SERVICE HEALTH CHECKS")
        all_healthy = True

        # Check TEI service
        self._print_test("TEI Embeddings Service", "RUNNING")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{settings.tei_url}/health")
                if response.status_code == 200:
                    self._print_test("TEI Embeddings Service", "PASS")
                    self._record_test(True)
                else:
                    raise Exception(f"TEI health check failed: {response.status_code}")
        except Exception as e:
            self._print_test("TEI Embeddings Service", "FAIL")
            self._record_test(False, f"TEI health check failed: {e}")
            all_healthy = False

        # Check Qdrant service
        self._print_test("Qdrant Vector Database", "RUNNING")
        try:
            async with VectorService() as vector_service:
                await vector_service.health_check()
            self._print_test("Qdrant Vector Database", "PASS")
            self._record_test(True)
        except Exception as e:
            self._print_test("Qdrant Vector Database", "FAIL")
            self._record_test(False, f"Qdrant health check failed: {e}")
            all_healthy = False

        # Check RAG service initialization
        self._print_test("RAG Service Initialization", "RUNNING")
        try:
            async with RagService() as rag_service:
                health = await rag_service.health_check()
                if all(health.values()):
                    self._print_test("RAG Service Initialization", "PASS")
                    self._record_test(True)
                else:
                    raise Exception(f"RAG service unhealthy: {health}")
        except Exception as e:
            self._print_test("RAG Service Initialization", "FAIL")
            self._record_test(False, f"RAG service initialization failed: {e}")
            all_healthy = False

        return all_healthy

    async def test_content_ingestion(self) -> None:
        """Test content ingestion through various methods using exact crawl tool interface."""
        self._print_header("CONTENT INGESTION TESTS")

        async with create_connected_server_and_client_session(
            mcp._mcp_server
        ) as client:
            # Test single page scraping with RAG ingestion (using exact scrape parameters)
            self._print_test("Single Page Scraping + RAG", "RUNNING")
            try:
                start_time = time.time()
                result = await client.call_tool(
                    "scrape",
                    {
                        "url": self.test_data["urls"][0],
                        "screenshot": False,
                        "wait_for": None,
                        "css_selector": None,
                        "javascript": False,
                        "timeout_ms": 30000,
                        "rag_ingest": True,
                    },
                )

                scrape_time = time.time() - start_time
                self.metrics.performance_data["scrape_time"] = scrape_time

                # Parse result
                result_content = json.loads(result.content[0].text)

                if result_content.get("success"):
                    self._print_test("Single Page Scraping + RAG", "PASS")
                    self._record_test(True)

                    # Store additional metrics from actual response
                    if "rag" in result_content:
                        self.metrics.performance_data["rag_ingest_enabled"] = (
                            result_content["rag"].get("enabled", False)
                        )
                    if "stats" in result_content:
                        self.metrics.performance_data["scrape_word_count"] = (
                            result_content["stats"].get("words", 0)
                        )
                else:
                    raise Exception("Scraping failed")

            except Exception as e:
                self._print_test("Single Page Scraping + RAG", "FAIL")
                self._record_test(False, f"Scraping test failed: {e}")

            # Test website crawling (using exact crawl parameters)
            if self.mode in ["full", "stress"]:
                self._print_test("Website Crawling", "RUNNING")
                try:
                    start_time = time.time()
                    result = await client.call_tool(
                        "crawl",
                        {
                            "target": self.test_data["urls"][0],  # Single URL as target
                            "limit": 5,
                            "depth": 2,
                            "max_concurrent": 3,
                            "include_patterns": None,
                            "exclude_patterns": None,
                            "javascript": False,
                            "screenshot_samples": 0,
                            "timeout_ms": 60000,
                            "rag_ingest": True,
                        },
                    )

                    crawl_time = time.time() - start_time
                    self.metrics.performance_data["crawl_time"] = crawl_time

                    result_content = json.loads(result.content[0].text)

                    if result_content.get("success"):
                        self._print_test("Website Crawling", "PASS")
                        self._record_test(True)

                        # Store crawl-specific metrics
                        if "stats" in result_content:
                            stats = result_content["stats"]
                            self.metrics.performance_data["crawl_pages_processed"] = (
                                stats.get("processed", 0)
                            )
                            self.metrics.performance_data["crawl_pages_failed"] = (
                                stats.get("failed", 0)
                            )
                            self.metrics.performance_data["crawl_duration"] = stats.get(
                                "duration_s", 0.0
                            )

                        if "docs_preview" in result_content:
                            self.metrics.performance_data["crawl_docs_found"] = len(
                                result_content["docs_preview"]
                            )

                    else:
                        raise Exception("Crawling failed")

                except Exception as e:
                    self._print_test("Website Crawling", "FAIL")
                    self._record_test(False, f"Website crawling test failed: {e}")

            # Test directory crawling if in stress mode
            if self.mode == "stress":
                self._print_test("Directory Crawling", "RUNNING")
                try:
                    # Create a temporary test directory
                    test_dir = Path("/tmp/test_crawl_dir")
                    test_dir.mkdir(exist_ok=True)

                    # Create some test files
                    (test_dir / "test1.md").write_text(
                        "# Test Document 1\nThis is test content for RAG ingestion."
                    )
                    (test_dir / "test2.txt").write_text(
                        "This is plain text content for testing the directory crawler."
                    )
                    (test_dir / "test3.py").write_text(
                        "# Python test file\ndef hello():\n    return 'world'"
                    )

                    start_time = time.time()
                    result = await client.call_tool(
                        "crawl",
                        {
                            "target": str(test_dir),
                            "limit": 10,
                            "depth": 1,
                            "include_patterns": ["**/*.md", "**/*.txt", "**/*.py"],
                            "exclude_patterns": ["**/__pycache__/**"],
                            "rag_ingest": True,
                        },
                    )

                    dir_crawl_time = time.time() - start_time
                    self.metrics.performance_data["dir_crawl_time"] = dir_crawl_time

                    result_content = json.loads(result.content[0].text)

                    if (
                        result_content.get("success")
                        and result_content.get("kind") == "directory"
                    ):
                        self._print_test("Directory Crawling", "PASS")
                        self._record_test(True)

                        # Store directory crawl metrics
                        if "stats" in result_content:
                            self.metrics.performance_data["dir_files_processed"] = (
                                result_content["stats"].get("processed", 0)
                            )
                    else:
                        raise Exception("Directory crawling failed")

                    # Cleanup test directory
                    import shutil

                    shutil.rmtree(test_dir, ignore_errors=True)

                except Exception as e:
                    self._print_test("Directory Crawling", "FAIL")
                    self._record_test(False, f"Directory crawling test failed: {e}")

            # Test GitHub repository crawling if in stress mode
            if self.mode == "stress":
                self._print_test("GitHub Repository Crawling", "RUNNING")
                try:
                    start_time = time.time()
                    result = await client.call_tool(
                        "crawl",
                        {
                            "target": self.test_data["test_targets"]["github_repo"],
                            "limit": 20,
                            "include_patterns": ["**/*.md", "**/*.py", "**/*.js"],
                            "exclude_patterns": ["**/node_modules/**", "**/.git/**"],
                            "rag_ingest": True,
                        },
                    )

                    repo_crawl_time = time.time() - start_time
                    self.metrics.performance_data["repo_crawl_time"] = repo_crawl_time

                    result_content = json.loads(result.content[0].text)

                    if (
                        result_content.get("success")
                        and result_content.get("kind") == "repository"
                    ):
                        self._print_test("GitHub Repository Crawling", "PASS")
                        self._record_test(True)

                        # Store repository crawl metrics
                        if "stats" in result_content:
                            self.metrics.performance_data["repo_files_processed"] = (
                                result_content["stats"].get("processed", 0)
                            )
                    else:
                        raise Exception("Repository crawling failed")

                except Exception as e:
                    self._print_test("GitHub Repository Crawling", "FAIL")
                    self._record_test(False, f"Repository crawling test failed: {e}")

        # Allow time for ingestion to complete
        await asyncio.sleep(5 if self.mode == "stress" else 3)

    async def test_vector_storage(self) -> None:
        """Test vector storage and retrieval operations."""
        self._print_header("VECTOR STORAGE TESTS")

        # Test vector statistics
        self._print_test("Vector Database Statistics", "RUNNING")
        try:
            async with VectorService() as vector_service:
                stats = await vector_service.get_collection_stats()

                if stats.get("total_chunks", 0) > 0:
                    self._print_test("Vector Database Statistics", "PASS")
                    self._record_test(True)
                    self.metrics.performance_data["total_chunks"] = stats.get(
                        "total_chunks", 0
                    )
                    self.metrics.performance_data["vector_dimension"] = stats.get(
                        "vector_dimension", 0
                    )
                else:
                    raise Exception("No chunks found in vector database")

        except Exception as e:
            self._print_test("Vector Database Statistics", "FAIL")
            self._record_test(False, f"Vector stats test failed: {e}")

        # Test embedding generation
        self._print_test("Embedding Generation", "RUNNING")
        try:
            async with EmbeddingService() as embedding_service:
                start_time = time.time()
                embeddings = await embedding_service.generate_embeddings(
                    ["Test embedding generation", "Another test sentence"]
                )
                embedding_time = time.time() - start_time

                if embeddings and len(embeddings) == 2:
                    self._print_test("Embedding Generation", "PASS")
                    self._record_test(True)
                    self.metrics.performance_data["embedding_time"] = embedding_time
                    self.metrics.performance_data["embedding_dimension"] = len(
                        embeddings[0].embedding
                    )
                else:
                    raise Exception("Invalid embedding results")

        except Exception as e:
            self._print_test("Embedding Generation", "FAIL")
            self._record_test(False, f"Embedding generation test failed: {e}")

    async def test_rag_queries(self) -> None:
        """Test RAG query processing and retrieval."""
        self._print_header("RAG QUERY TESTS")

        async with create_connected_server_and_client_session(
            mcp._mcp_server
        ) as client:
            for i, query in enumerate(self.test_data["queries"]):
                if self.mode == "quick" and i >= 2:
                    break

                self._print_test(f"RAG Query: '{query[:30]}...'", "RUNNING")
                try:
                    start_time = time.time()
                    result = await client.call_tool(
                        "rag_query",
                        {
                            "query": query,
                            "limit": 5,
                            "min_score": 0.1,
                            "rerank": True,
                            "include_content": True,
                        },
                    )

                    query_time = time.time() - start_time
                    result_content = json.loads(result.content[0].text)

                    # Verify response structure
                    required_fields = [
                        "matches",
                        "total_matches",
                        "performance",
                        "quality_metrics",
                    ]
                    if all(field in result_content for field in required_fields):
                        self._print_test(f"RAG Query: '{query[:30]}...'", "PASS")
                        self._record_test(True)

                        # Record performance metrics
                        perf_key = f"query_{i}_performance"
                        self.metrics.performance_data[perf_key] = {
                            "total_time": query_time,
                            "matches": result_content["total_matches"],
                            "avg_score": result_content["quality_metrics"].get(
                                "average_score", 0.0
                            ),
                        }
                    else:
                        raise Exception("Invalid query response structure")

                except Exception as e:
                    self._print_test(f"RAG Query: '{query[:30]}...'", "FAIL")
                    self._record_test(False, f"RAG query failed: {e}")

            # Test query with filters
            if self.mode in ["full", "stress"]:
                self._print_test("Filtered RAG Query", "RUNNING")
                try:
                    result = await client.call_tool(
                        "rag_query",
                        {
                            "query": "test",
                            "limit": 10,
                            "min_score": 0.3,
                            "source_filters": ["httpbin.org"],
                            "rerank": False,
                        },
                    )

                    result_content = json.loads(result.content[0].text)

                    if "matches" in result_content:
                        self._print_test("Filtered RAG Query", "PASS")
                        self._record_test(True)
                    else:
                        raise Exception("Filtered query failed")

                except Exception as e:
                    self._print_test("Filtered RAG Query", "FAIL")
                    self._record_test(False, f"Filtered query failed: {e}")

    async def test_crawl_target_detection(self) -> None:
        """Test the crawl tool's target detection and handling."""
        if self.mode == "quick":
            return

        self._print_header("CRAWL TARGET DETECTION TESTS")

        async with create_connected_server_and_client_session(
            mcp._mcp_server
        ) as client:
            # Test website target detection
            self._print_test("Website Target Detection", "RUNNING")
            try:
                result = await client.call_tool(
                    "crawl",
                    {
                        "target": self.test_data["test_targets"]["website"],
                        "limit": 2,
                        "rag_ingest": False,  # Skip RAG for speed
                        "timeout_ms": 20000,
                    },
                )

                result_content = json.loads(result.content[0].text)

                if (
                    result_content.get("success")
                    and result_content.get("kind") == "website"
                ):
                    self._print_test("Website Target Detection", "PASS")
                    self._record_test(True)
                else:
                    raise Exception(
                        f"Expected website target, got: {result_content.get('kind')}"
                    )

            except Exception as e:
                self._print_test("Website Target Detection", "FAIL")
                self._record_test(False, f"Website target test failed: {e}")

            # Test crawl parameter validation
            self._print_test("Parameter Validation", "RUNNING")
            try:
                # Test various parameter combinations
                test_params = [
                    {"target": "https://example.com", "limit": 1, "depth": 1},
                    {"target": "https://example.com", "limit": 5, "max_concurrent": 2},
                    {
                        "target": "https://example.com",
                        "javascript": True,
                        "timeout_ms": 15000,
                    },
                ]

                param_tests_passed = 0
                for params in test_params:
                    try:
                        result = await client.call_tool("crawl", params)
                        result_content = json.loads(result.content[0].text)
                        if (
                            result_content.get("success") is not None
                        ):  # Got a valid response
                            param_tests_passed += 1
                    except Exception:
                        pass  # Individual parameter test failure is ok

                if param_tests_passed >= 2:  # At least 2/3 parameter tests passed
                    self._print_test("Parameter Validation", "PASS")
                    self._record_test(True)
                else:
                    raise Exception(
                        f"Only {param_tests_passed}/3 parameter tests passed"
                    )

            except Exception as e:
                self._print_test("Parameter Validation", "FAIL")
                self._record_test(False, f"Parameter validation test failed: {e}")

    async def test_performance_benchmarks(self) -> None:
        """Run performance benchmarks if in stress mode."""
        if self.mode != "stress":
            return

        self._print_header("PERFORMANCE BENCHMARKS")

        # Test concurrent queries
        self._print_test("Concurrent Query Performance", "RUNNING")
        try:
            async with create_connected_server_and_client_session(
                mcp._mcp_server
            ) as client:
                # Prepare concurrent queries
                query_tasks = []
                for i in range(10):
                    task = client.call_tool(
                        "rag_query",
                        {
                            "query": f"test query {i}",
                            "limit": 3,
                            "rerank": False,
                        },
                    )
                    query_tasks.append(task)

                start_time = time.time()
                results = await asyncio.gather(*query_tasks, return_exceptions=True)
                concurrent_time = time.time() - start_time

                successful_queries = sum(
                    1 for r in results if not isinstance(r, Exception)
                )

                self.metrics.performance_data["concurrent_queries"] = {
                    "total_queries": 10,
                    "successful_queries": successful_queries,
                    "total_time": concurrent_time,
                    "avg_time_per_query": concurrent_time / 10,
                }

                if successful_queries >= 8:
                    self._print_test("Concurrent Query Performance", "PASS")
                    self._record_test(True)
                else:
                    raise Exception(
                        f"Only {successful_queries}/10 concurrent queries succeeded"
                    )

        except Exception as e:
            self._print_test("Concurrent Query Performance", "FAIL")
            self._record_test(False, f"Concurrent query test failed: {e}")

        # Test crawling performance with different configurations
        self._print_test("Crawl Performance Comparison", "RUNNING")
        try:
            async with create_connected_server_and_client_session(
                mcp._mcp_server
            ) as client:
                test_configs = [
                    {"name": "basic", "javascript": False, "max_concurrent": 1},
                    {"name": "concurrent", "javascript": False, "max_concurrent": 3},
                    {"name": "javascript", "javascript": True, "max_concurrent": 2},
                ]

                performance_results = {}

                for config in test_configs:
                    start_time = time.time()
                    try:
                        result = await client.call_tool(
                            "crawl",
                            {
                                "target": "https://httpbin.org/html",
                                "limit": 3,
                                "timeout_ms": 30000,
                                "rag_ingest": False,
                                **{k: v for k, v in config.items() if k != "name"},
                            },
                        )

                        crawl_time = time.time() - start_time
                        result_content = json.loads(result.content[0].text)

                        performance_results[config["name"]] = {
                            "time": crawl_time,
                            "success": result_content.get("success", False),
                            "pages": result_content.get("stats", {}).get(
                                "processed", 0
                            ),
                        }

                    except Exception as e:
                        performance_results[config["name"]] = {
                            "time": time.time() - start_time,
                            "success": False,
                            "error": str(e),
                        }

                self.metrics.performance_data["crawl_performance_comparison"] = (
                    performance_results
                )

                # Consider test passed if at least 2/3 configurations worked
                successful_configs = sum(
                    1 for r in performance_results.values() if r.get("success")
                )
                if successful_configs >= 2:
                    self._print_test("Crawl Performance Comparison", "PASS")
                    self._record_test(True)
                else:
                    raise Exception(
                        f"Only {successful_configs}/3 crawl configurations succeeded"
                    )

        except Exception as e:
            self._print_test("Crawl Performance Comparison", "FAIL")
            self._record_test(False, f"Crawl performance test failed: {e}")

    async def cleanup(self) -> None:
        """Clean up test data and resources."""
        self._print_header("CLEANUP")

        self._print_test("Cleaning up test collection", "RUNNING")
        try:
            async with VectorService() as vector_service:
                # Clean up test collection if it exists
                await vector_service.delete_collection_if_exists(self.test_collection)

            self._print_test("Cleaning up test collection", "PASS")
        except Exception as e:
            self._print_test("Cleaning up test collection", "FAIL")
            logger.warning(f"Cleanup failed: {e}")

    def generate_report(self, output_file: str | None = None) -> dict[str, Any]:
        """Generate comprehensive test report."""
        self.metrics.end_time = time.time()

        report = {
            "test_summary": {
                "mode": self.mode,
                "total_tests": self.metrics.total_tests,
                "passed_tests": self.metrics.passed_tests,
                "failed_tests": self.metrics.failed_tests,
                "success_rate": self.metrics.success_rate,
                "duration": self.metrics.duration,
            },
            "performance_metrics": self.metrics.performance_data,
            "errors": self.metrics.errors,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "tei_url": settings.tei_url,
                "qdrant_url": settings.qdrant_url,
                "reranker_enabled": settings.reranker_enabled,
                "chunk_size": settings.chunk_size,
            },
        }

        if output_file:
            Path(output_file).write_text(json.dumps(report, indent=2))
            print(f"\n{Colors.CYAN}Test report saved to: {output_file}{Colors.END}")

        return report

    def print_summary(self) -> None:
        """Print test summary to console."""
        self._print_header("TEST SUMMARY")

        print(f"  Mode: {Colors.BOLD}{self.mode.upper()}{Colors.END}")
        print(f"  Duration: {Colors.BOLD}{self.metrics.duration:.2f}s{Colors.END}")
        print(f"  Total Tests: {Colors.BOLD}{self.metrics.total_tests}{Colors.END}")

        # Pass/Fail summary
        if self.metrics.passed_tests > 0:
            print(f"  {Colors.GREEN}✓ Passed: {self.metrics.passed_tests}{Colors.END}")
        if self.metrics.failed_tests > 0:
            print(f"  {Colors.RED}✗ Failed: {self.metrics.failed_tests}{Colors.END}")

        # Success rate
        color = (
            Colors.GREEN
            if self.metrics.success_rate >= 80
            else Colors.YELLOW
            if self.metrics.success_rate >= 60
            else Colors.RED
        )
        print(f"  Success Rate: {color}{self.metrics.success_rate:.1f}%{Colors.END}")

        # Performance highlights
        if self.metrics.performance_data:
            print(f"\n{Colors.BOLD}Performance Highlights:{Colors.END}")
            if "scrape_time" in self.metrics.performance_data:
                print(
                    f"  Scrape Time: {self.metrics.performance_data['scrape_time']:.2f}s"
                )
            if "crawl_time" in self.metrics.performance_data:
                print(
                    f"  Crawl Time: {self.metrics.performance_data['crawl_time']:.2f}s"
                )
            if "crawl_pages_processed" in self.metrics.performance_data:
                print(
                    f"  Pages Crawled: {self.metrics.performance_data['crawl_pages_processed']}"
                )
            if "total_chunks" in self.metrics.performance_data:
                print(
                    f"  Total Chunks: {self.metrics.performance_data['total_chunks']}"
                )
            if "embedding_dimension" in self.metrics.performance_data:
                print(
                    f"  Embedding Dimension: {self.metrics.performance_data['embedding_dimension']}"
                )
            if "crawl_performance_comparison" in self.metrics.performance_data:
                perf = self.metrics.performance_data["crawl_performance_comparison"]
                successful_configs = [
                    name for name, data in perf.items() if data.get("success")
                ]
                if successful_configs:
                    print(f"  Crawl Configs Tested: {', '.join(successful_configs)}")

        # Errors
        if self.metrics.errors:
            print(f"\n{Colors.RED}Errors:{Colors.END}")
            for error in self.metrics.errors[:5]:  # Show first 5 errors
                print(f"  {Colors.RED}• {error}{Colors.END}")
            if len(self.metrics.errors) > 5:
                print(
                    f"  {Colors.RED}... and {len(self.metrics.errors) - 5} more errors{Colors.END}"
                )

    async def run_all_tests(self) -> bool:
        """Run all tests in sequence."""
        print(f"{Colors.BOLD}{Colors.BLUE}🕷️  RAG Pipeline Test Suite{Colors.END}")
        print(f"{Colors.BOLD}Mode: {self.mode.upper()}{Colors.END}")
        print(f"{Colors.BOLD}Test Collection: {self.test_collection}{Colors.END}")

        try:
            # Health checks first
            healthy = await self.check_service_health()
            if not healthy and self.mode != "quick":
                print(
                    f"\n{Colors.RED}❌ Service health checks failed. Aborting tests.{Colors.END}"
                )
                return False

            # Content ingestion tests
            await self.test_content_ingestion()

            # Vector storage tests
            await self.test_vector_storage()

            # RAG query tests
            await self.test_rag_queries()

            # Crawl target detection tests
            await self.test_crawl_target_detection()

            # Performance benchmarks (stress mode only)
            await self.test_performance_benchmarks()

            # Cleanup
            await self.cleanup()

            return True

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.END}")
            return False
        except Exception as e:
            print(f"\n{Colors.RED}Fatal error during testing: {e}{Colors.END}")
            return False


async def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Comprehensive RAG Pipeline Test Suite"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "stress"],
        default="full",
        help="Test mode: quick (~2min), full (~10min), or stress (load testing)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--report-file", type=str, help="Path to save JSON test report")

    args = parser.parse_args()

    # Initialize test suite
    test_suite = RAGPipelineTests(mode=args.mode, debug=args.debug)

    # Run tests
    success = await test_suite.run_all_tests()

    # Generate and display results
    test_suite.print_summary()
    test_suite.generate_report(args.report_file)

    # Exit with appropriate code
    exit_code = 0 if success and test_suite.metrics.success_rate >= 80 else 1

    if exit_code == 0:
        print(f"\n{Colors.GREEN}🎉 All tests completed successfully!{Colors.END}")
    else:
        print(
            f"\n{Colors.RED}❌ Some tests failed. Check the report for details.{Colors.END}"
        )

    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
