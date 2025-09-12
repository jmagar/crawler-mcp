"""
Performance monitoring and hooks for optimized high-performance web crawler.

This module provides comprehensive monitoring capabilities including metrics collection,
performance tracking, and customizable hooks for the crawling process.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil


@dataclass
class CrawlMetrics:
    """Container for crawl performance metrics"""

    # Basic counters
    pages_crawled: int = 0
    pages_failed: int = 0
    hash_placeholders_detected: int = 0
    urls_discovered: int = 0

    # Timing metrics
    start_time: float | None = None
    end_time: float | None = None
    total_duration: float = 0.0
    pages_per_second: float = 0.0

    # Content metrics
    total_content_bytes: int = 0
    average_content_length: float = 0.0
    min_content_length: int = 0
    max_content_length: int = 0

    # System metrics
    peak_memory_usage_mb: float = 0.0
    average_cpu_usage: float = 0.0
    average_process_cpu_usage: float = 0.0
    concurrent_sessions_peak: int = 0

    # Error tracking
    error_counts: dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0

    # Quality metrics
    content_quality_score: float = 0.0
    duplicate_content_detected: int = 0

    # Validation diagnostics
    invalid_reason_counts: dict[str, int] = field(default_factory=dict)
    relaxed_acceptances: int = 0
    relaxed_acceptance_reason_counts: dict[str, int] = field(default_factory=dict)
    content_validations_recorded: int = 0


class PerformanceMonitor:
    """Monitor crawler performance and provide customizable hooks"""

    def __init__(self, config: object | None = None):
        """
        Initialize performance monitor.

        Args:
            config: Optional optimized crawler configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Metrics tracking
        self.metrics = CrawlMetrics()
        self.hooks: defaultdict[str, list[Callable]] = defaultdict(list)

        # Performance tracking
        self._performance_samples: deque = deque(maxlen=100)  # Last 100 samples
        self._system_metrics: deque = deque(maxlen=50)  # System metrics
        self._proc = None  # psutil.Process primed at start

        # State tracking
        self._crawl_active = False
        self._monitoring_task: asyncio.Task | None = None
        self._hook_queue: asyncio.Queue | None = None
        self._hook_worker_task: asyncio.Task | None = None
        # Error/invalid samples
        self._error_samples: dict[str, list[str]] = {}
        self._invalid_reason_samples: dict[str, list[str]] = {}
        # Content duplicate tracking (hash -> info)
        self._content_hashes: dict[str, dict[str, Any]] = {}
        # Completion timestamps for peak concurrency estimate
        from collections import deque as _dq  # local alias

        self._completion_times: _dq[float] = _dq(maxlen=5000)

        # Hook types supported
        self.supported_hooks = {
            "crawl_started",
            "crawl_completed",
            "url_discovered",
            "page_crawled",
            "page_failed",
            "hash_placeholder_detected",
            "content_validated",
            "performance_sample",
            "system_alert",
            "crawl_paused",
            "crawl_resumed",
        }

        # Embeddings stats (populated by strategy when enabled)
        self._embeddings_stats: dict[str, Any] = {}
        self._vectorstore_stats: dict[str, Any] = {}

    # Validation diagnostics API
    def record_invalid_reason(self, reason: str, url: str | None = None) -> None:
        self.metrics.invalid_reason_counts[reason] = (
            self.metrics.invalid_reason_counts.get(reason, 0) + 1
        )
        if url:
            lst = self._invalid_reason_samples.setdefault(reason, [])
            if len(lst) < 5 and url not in lst:
                lst.append(url)

    def record_relaxed_acceptance(self, reason: str) -> None:
        self.metrics.relaxed_acceptances += 1
        self.metrics.relaxed_acceptance_reason_counts[reason] = (
            self.metrics.relaxed_acceptance_reason_counts.get(reason, 0) + 1
        )

    def register_hook(self, hook_type: str, hook_func: Callable[..., Any]) -> None:
        """
        Register a custom hook for monitoring events.

        Args:
            hook_type: Type of hook to register
            hook_func: Function to call when hook is triggered

        Raises:
            ValueError: If hook_type is not supported
        """
        if hook_type not in self.supported_hooks:
            raise ValueError(
                f"Unsupported hook type: {hook_type}. "
                f"Supported: {', '.join(self.supported_hooks)}"
            )

        self.hooks[hook_type].append(hook_func)
        self.logger.debug("Registered hook for %s", hook_type)

    def unregister_hook(self, hook_type: str, hook_func: Callable[..., Any]) -> bool:
        """
        Unregister a hook.

        Args:
            hook_type: Type of hook to unregister
            hook_func: Function to unregister

        Returns:
            True if hook was found and removed
        """
        if hook_type in self.hooks:
            try:
                self.hooks[hook_type].remove(hook_func)
                return True
            except ValueError:
                pass
        return False

    async def trigger_hook(self, hook_type: str, **kwargs) -> None:
        """
        Trigger all hooks of a specific type.

        Args:
            hook_type: Type of hook to trigger
            **kwargs: Arguments to pass to hook functions
        """
        if hook_type not in self.hooks:
            return

        for hook_func in self.hooks[hook_type]:
            try:
                if asyncio.iscoroutinefunction(hook_func):
                    await hook_func(**kwargs)
                else:
                    hook_func(**kwargs)
            except Exception as e:
                self.logger.error(f"Hook {hook_type} failed: {e}")

    def start_crawl_monitoring(self, urls: list[str]) -> None:
        """
        Start monitoring a crawl session.

        Args:
            urls: List of URLs being crawled
        """
        self.metrics = CrawlMetrics()  # Reset metrics
        self.metrics.start_time = time.time()
        self.metrics.urls_discovered = len(urls)
        self._crawl_active = True

        # Initialize hook queue/worker
        self._start_hook_worker()

        # Prime CPU counters so the first sample is meaningful
        try:
            import psutil as _ps

            self._proc = _ps.Process()
            # prime both process and system
            _ = self._proc.cpu_percent(interval=None)
            _ = _ps.cpu_percent(interval=None)
        except Exception:
            self._proc = None

        # Start system monitoring task (always on; lightweight 1s sampling)
        self._monitoring_task = asyncio.create_task(self._system_monitoring_loop())

        self.logger.info(f"Started crawl monitoring for {len(urls)} URLs")
        self._enqueue_hook("crawl_started", urls=urls, metrics=self.metrics)

    def record_page_success(
        self, url: str, content_length: int, crawl_time: float
    ) -> None:
        """
        Record a successful page crawl.

        Args:
            url: URL that was crawled
            content_length: Length of extracted content
            crawl_time: Time taken to crawl the page
        """
        self.metrics.pages_crawled += 1
        self.metrics.total_content_bytes += content_length

        # Update content length statistics
        if self.metrics.pages_crawled == 1:
            self.metrics.min_content_length = content_length
            self.metrics.max_content_length = content_length
        else:
            self.metrics.min_content_length = min(
                self.metrics.min_content_length, content_length
            )
            self.metrics.max_content_length = max(
                self.metrics.max_content_length, content_length
            )

        # Calculate average content length
        self.metrics.average_content_length = (
            self.metrics.total_content_bytes / self.metrics.pages_crawled
        )

        # Record performance sample
        self._performance_samples.append(
            {
                "timestamp": time.time(),
                "url": url,
                "content_length": content_length,
                "crawl_time": crawl_time,
            }
        )

        # Update pages per second
        if self.metrics.start_time:
            elapsed = time.time() - self.metrics.start_time
            self.metrics.pages_per_second = self.metrics.pages_crawled / elapsed

        # Track completion times and update peak concurrency estimate
        now = time.time()
        self._completion_times.append(now)
        # count completions within last 1s window
        try:
            cutoff = now - 1.0
            recent = 0
            for ts in reversed(self._completion_times):
                if ts >= cutoff:
                    recent += 1
                else:
                    break
            if recent > (self.metrics.concurrent_sessions_peak or 0):
                self.metrics.concurrent_sessions_peak = recent
        except Exception:
            pass

        self._enqueue_hook(
            "page_crawled",
            url=url,
            content_length=content_length,
            crawl_time=crawl_time,
            metrics=self.metrics,
        )

    def record_page_failure(self, url: str, error: str) -> None:
        """
        Record a failed page crawl.

        Args:
            url: URL that failed
            error: Error message
        """
        self.metrics.pages_failed += 1

        # Track error types
        error_type = self._classify_error(error)
        self.metrics.error_counts[error_type] = (
            self.metrics.error_counts.get(error_type, 0) + 1
        )

        # Calculate error rate
        total_attempts = self.metrics.pages_crawled + self.metrics.pages_failed
        if total_attempts > 0:
            self.metrics.error_rate = self.metrics.pages_failed / total_attempts

        self.logger.debug("Recorded failure for %s: %s", url, error_type)
        self._enqueue_hook(
            "page_failed",
            url=url,
            error=error,
            error_type=error_type,
            metrics=self.metrics,
        )
        # keep small sample per error type
        try:
            lst = self._error_samples.setdefault(error_type, [])
            if len(lst) < 5 and url not in lst:
                lst.append(url)
        except Exception:
            pass

    def record_hash_placeholder(self, url: str) -> None:
        """
        Record detection of a hash placeholder.

        Args:
            url: URL where hash placeholder was detected
        """
        self.metrics.hash_placeholders_detected += 1

        self.logger.warning(f"Hash placeholder detected: {url}")
        self._enqueue_hook("hash_placeholder_detected", url=url, metrics=self.metrics)

    def record_content_validation(
        self, url: str, quality_score: float, is_duplicate: bool = False
    ) -> None:
        """
        Record content validation results.

        Args:
            url: URL that was validated
            quality_score: Content quality score (0.0-1.0)
            is_duplicate: Whether content was detected as duplicate
        """
        # Update quality metrics
        total_validated = self.metrics.content_validations_recorded
        if total_validated > 0:
            # Running average of quality scores
            current_avg = self.metrics.content_quality_score
            self.metrics.content_quality_score = (
                current_avg * total_validated + quality_score
            ) / (total_validated + 1)
        else:
            # First validation
            self.metrics.content_quality_score = quality_score
        # Track validations recorded
        self.metrics.content_validations_recorded += 1

        if is_duplicate:
            self.metrics.duplicate_content_detected += 1

        self._enqueue_hook(
            "content_validated",
            url=url,
            quality_score=quality_score,
            is_duplicate=is_duplicate,
            metrics=self.metrics,
        )

    def record_content_hash(self, url: str, content: str) -> bool:
        """Record content hash for duplicate detection. Returns True if duplicate."""
        try:
            import hashlib

            text = (content or "").strip()
            if not text:
                return False
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()
            info = self._content_hashes.get(h)
            if info is None:
                self._content_hashes[h] = {
                    "count": 1,
                    "first_url": url,
                    "sample_urls": [url],
                }
                return False
            # Already seen
            info["count"] += 1
            if len(info["sample_urls"]) < 5 and url not in info["sample_urls"]:
                info["sample_urls"].append(url)
            return True
        except Exception:
            return False

    def finish_crawl_monitoring(self) -> CrawlMetrics:
        """
        Finish monitoring and return final metrics.

        Returns:
            Final crawl metrics
        """
        self.metrics.end_time = time.time()
        if self.metrics.start_time:
            self.metrics.total_duration = (
                self.metrics.end_time - self.metrics.start_time
            )

            # Final pages per second calculation
            if self.metrics.total_duration > 0:
                self.metrics.pages_per_second = (
                    self.metrics.pages_crawled / self.metrics.total_duration
                )

        self._crawl_active = False

        # Stop system monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None

        self.logger.info(
            f"Crawl monitoring completed: {self.metrics.pages_crawled} pages, "
            f"{self.metrics.pages_per_second:.2f} pages/sec, "
            f"{self.metrics.pages_failed} failures"
        )

        self._enqueue_hook("crawl_completed", metrics=self.metrics)

        # Stop hook worker
        self._stop_hook_worker()

        return self.metrics

    async def _system_monitoring_loop(self) -> None:
        """Background task for system metrics monitoring"""
        try:
            while self._crawl_active:
                await self._collect_system_metrics()
                await asyncio.sleep(1.0)  # Sample every second
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"System monitoring failed: {e}")

    async def _collect_system_metrics(self) -> None:
        """Collect current process and system metrics"""
        try:
            # Process-focused metrics for adaptive tuning
            proc = self._proc or psutil.Process()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            # Instantaneous CPU deltas since last call (primed on start)
            cpu_proc = float(proc.cpu_percent(interval=None))
            cpu_sys = float(psutil.cpu_percent(interval=None))

            # System snapshot for alerts & context
            sys_mem = psutil.virtual_memory()

            # Peak process memory
            if rss_mb > self.metrics.peak_memory_usage_mb:
                self.metrics.peak_memory_usage_mb = rss_mb

            # Rolling sample buffer
            self._system_metrics.append(
                {
                    "timestamp": time.time(),
                    "memory_mb": rss_mb,
                    "memory_percent": sys_mem.percent,
                    "system_cpu_percent": cpu_sys,
                    "process_cpu_percent": cpu_proc,
                }
            )

            # Average CPU over recent samples
            if len(self._system_metrics) > 0:
                total = 0.0
                total_proc = 0.0
                count = 0
                for m in list(self._system_metrics)[-10:]:
                    total += float(m.get("system_cpu_percent", 0.0))
                    total_proc += float(m.get("process_cpu_percent", 0.0))
                    count += 1
                if count:
                    self.metrics.average_cpu_usage = total / count
                    self.metrics.average_process_cpu_usage = total_proc / count

            # Alerts on system pressure
            if sys_mem.percent > 90:
                await self.trigger_hook(
                    "system_alert",
                    alert_type="high_memory",
                    memory_percent=sys_mem.percent,
                )
            if cpu_sys > 95:
                await self.trigger_hook(
                    "system_alert", alert_type="high_cpu", cpu_percent=cpu_sys
                )

            # Emit sample
            self._enqueue_hook(
                "performance_sample",
                memory_mb=rss_mb,
                memory_percent=sys_mem.percent,
                cpu_percent=cpu_sys,
                pages_per_second=self.metrics.pages_per_second,
                metrics=self.metrics,
            )

        except Exception:
            self.logger.debug("Failed to collect system metrics", exc_info=True)

    def _classify_error(self, error: str) -> str:
        """Classify error type for tracking"""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "permission" in error_lower or "403" in error_lower:
            return "permission"
        elif "404" in error_lower or "not found" in error_lower:
            return "not_found"
        elif "500" in error_lower or "502" in error_lower or "503" in error_lower:
            return "server_error"
        elif "memory" in error_lower:
            return "memory"
        elif "javascript" in error_lower:
            return "javascript"
        else:
            return "other"

    def get_performance_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive performance report.

        Returns:
            Dictionary with performance metrics and analysis
        """

        # helpers
        def _fmt_bytes(n: int | float) -> str:
            try:
                n = float(n)
            except Exception:
                return str(n)
            units = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            while n >= 1024 and i < len(units) - 1:
                n /= 1024.0
                i += 1
            return f"{n:.2f} {units[i]}"

        avg_size = (
            self.metrics.total_content_bytes / self.metrics.pages_crawled
            if self.metrics.pages_crawled > 0
            else 0
        )

        report = {
            "summary": {
                "pages_crawled": self.metrics.pages_crawled,
                "pages_failed": self.metrics.pages_failed,
                "success_rate": (
                    self.metrics.pages_crawled
                    / (self.metrics.pages_crawled + self.metrics.pages_failed)
                    if (self.metrics.pages_crawled + self.metrics.pages_failed) > 0
                    else 0
                ),
                "pages_per_second": self.metrics.pages_per_second,
                "total_duration": self.metrics.total_duration,
                "total_content_human": _fmt_bytes(self.metrics.total_content_bytes),
                "average_page_size_human": _fmt_bytes(avg_size),
                "content_quality_score": self.metrics.content_quality_score,
            },
            "validation_summary": {
                "invalid_reasons": dict(self.metrics.invalid_reason_counts),
                "relaxed_acceptances_total": self.metrics.relaxed_acceptances,
                "relaxed_acceptance_reasons": dict(
                    self.metrics.relaxed_acceptance_reason_counts
                ),
                "content_validations_recorded": self.metrics.content_validations_recorded,
                "invalid_reason_samples": {
                    k: v[:] for k, v in self._invalid_reason_samples.items()
                },
            },
            "content_analysis": {
                "total_content_bytes": self.metrics.total_content_bytes,
                "total_content_human": _fmt_bytes(self.metrics.total_content_bytes),
                "average_content_length": self.metrics.average_content_length,
                "average_content_human": _fmt_bytes(
                    self.metrics.average_content_length
                ),
                "content_range": {
                    "min": self.metrics.min_content_length,
                    "max": self.metrics.max_content_length,
                },
                "quality_score": self.metrics.content_quality_score,
                "hash_placeholders": self.metrics.hash_placeholders_detected,
                "duplicates": self.metrics.duplicate_content_detected,
                "duplicate_groups": [
                    {
                        "first_url": v.get("first_url"),
                        "count": v.get("count", 0),
                        "sample_urls": v.get("sample_urls", [])[:5],
                    }
                    for _, v in sorted(
                        self._content_hashes.items(),
                        key=lambda it: -int(it[1].get("count", 0)),
                    )
                    if int(v.get("count", 0)) > 1
                ][:5],
            },
            "system_performance": {
                "peak_memory_mb": self.metrics.peak_memory_usage_mb,
                "average_cpu_usage": self.metrics.average_cpu_usage,
                "process_cpu_avg": self.metrics.average_process_cpu_usage,
                "concurrent_sessions_peak": self.metrics.concurrent_sessions_peak,
            },
            "error_analysis": {
                "total_errors": self.metrics.pages_failed,
                "error_rate": self.metrics.error_rate,
                "error_breakdown": dict(self.metrics.error_counts),
                "error_samples": {k: v[:] for k, v in self._error_samples.items()},
            },
            "performance_trend": self._analyze_performance_trend(),
            "recommendations": self._generate_recommendations(),
        }

        if self._embeddings_stats:
            # Keep only simple JSON-serializable fields
            report["embeddings"] = dict(self._embeddings_stats)

        if self._vectorstore_stats:
            report["vector_store"] = dict(self._vectorstore_stats)

        return report

    # Embeddings
    def record_embeddings_stats(self, **stats: Any) -> None:
        """Attach TEI embeddings summary to the performance report."""
        from contextlib import suppress

        with suppress(Exception):
            self._embeddings_stats.update(stats)

    def record_vectorstore_stats(self, **stats: Any) -> None:
        """Attach vector store (Qdrant) upsert summary to the report."""
        from contextlib import suppress

        with suppress(Exception):
            self._vectorstore_stats.update(stats)

    def _analyze_performance_trend(self) -> dict[str, Any]:
        """Analyze performance trends from samples"""
        if len(self._performance_samples) < 5:
            return {"status": "insufficient_data"}

        # Compute recent rate over a stable time window to avoid spikes
        now = time.time()
        window_s = 5.0
        window_samples = [
            s for s in self._performance_samples if (now - s["timestamp"]) <= window_s
        ]

        if window_samples:
            earliest = min(s["timestamp"] for s in window_samples)
            time_span = max(1.0, now - earliest)  # clamp to >=1s for stability
            recent_rate = len(window_samples) / time_span
        else:
            recent_rate = 0.0

        # Compare with overall rate
        overall_rate = self.metrics.pages_per_second
        trend = "stable"

        if recent_rate > overall_rate * 1.2:
            trend = "improving"
        elif recent_rate < overall_rate * 0.8:
            trend = "declining"

        return {
            "status": "analyzed",
            "trend": trend,
            "recent_rate": recent_rate,
            "overall_rate": overall_rate,
            "sample_count": len(window_samples),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate performance recommendations"""
        recommendations = []

        # Error rate recommendations
        if self.metrics.error_rate > 0.1:
            recommendations.append(
                "High error rate detected. Consider reducing concurrency or adding retry logic."
            )

        # Hash placeholder recommendations
        if self.metrics.hash_placeholders_detected > 0:
            recommendations.append(
                "Hash placeholders detected. Review content extraction configuration."
            )

        # Performance recommendations
        if self.metrics.pages_per_second < 5.0:
            recommendations.append(
                "Low pages per second. Consider increasing concurrency or optimizing browser config."
            )

        # Memory recommendations
        if self.metrics.peak_memory_usage_mb > 8000:  # 8GB
            recommendations.append(
                "High memory usage detected. Consider reducing concurrency or batch size."
            )

        # Content quality recommendations (only if we recorded any validations)
        if (
            self.metrics.content_validations_recorded > 0
            and self.metrics.content_quality_score < 0.7
        ):
            recommendations.append(
                "Low content quality scores. Review content extraction and filtering settings."
            )

        return recommendations

    # Internal: hook worker
    def _start_hook_worker(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            if self._hook_queue is None:
                self._hook_queue = asyncio.Queue()
            if self._hook_worker_task is None or self._hook_worker_task.done():
                self._hook_worker_task = loop.create_task(self._hook_worker())
        except RuntimeError:
            # No event loop running, cannot start worker
            pass

    def _stop_hook_worker(self) -> None:
        # Signal termination and cancel worker
        if self._hook_queue is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._hook_queue.put_nowait((None, {}))  # sentinel
        if self._hook_worker_task:
            self._hook_worker_task.cancel()
            self._hook_worker_task = None

    def _enqueue_hook(self, hook_type: str | None, **kwargs) -> None:
        if not hook_type:
            return
        if self._hook_queue is None:
            # Fallback to direct call if queue not started
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.trigger_hook(hook_type, **kwargs))  # noqa: RUF006
            except RuntimeError:
                # No event loop running, schedule for later
                pass
            return
        try:
            self._hook_queue.put_nowait((hook_type, kwargs))
        except Exception:
            # In case of full queue or other errors, fallback to direct execution
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.trigger_hook(hook_type, **kwargs))
            except RuntimeError:
                # No event loop running, skip hook execution
                pass

    async def _hook_worker(self) -> None:
        try:
            while True:
                item = await self._hook_queue.get()
                if not item:
                    continue
                hook_type, payload = item
                if hook_type is None:
                    break  # sentinel
                try:
                    await self.trigger_hook(hook_type, **payload)
                except Exception:
                    self.logger.debug(
                        "Hook worker error for %s", hook_type, exc_info=True
                    )
                finally:
                    self._hook_queue.task_done()
        except asyncio.CancelledError:
            pass
