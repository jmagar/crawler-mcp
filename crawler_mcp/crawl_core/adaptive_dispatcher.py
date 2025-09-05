"""
Adaptive dispatcher tuning based on PerformanceMonitor metrics.

This module provides a small helper that adjusts MemoryAdaptiveDispatcher
concurrency parameters at runtime using observed CPU, error rate, and throughput.
"""

from __future__ import annotations

import asyncio
import logging

from crawl4ai import MemoryAdaptiveDispatcher

from crawler_mcp.utils.monitoring import PerformanceMonitor


class ConcurrencyTuner:
    """Adjust MemoryAdaptiveDispatcher concurrency using live metrics."""

    def __init__(
        self,
        dispatcher: MemoryAdaptiveDispatcher,
        monitor: PerformanceMonitor,
        min_concurrency: int = 2,
        max_concurrency: int | None = None,
        sample_interval: float = 2.0,
    ) -> None:
        self.dispatcher = dispatcher
        self.monitor = monitor
        self.min_concurrency = max(1, int(min_concurrency))
        # Default upper bound to initial max_session_permit if not provided
        self.max_concurrency = (
            int(getattr(dispatcher, "max_session_permit", 8))
            if max_concurrency is None
            else int(max_concurrency)
        )
        self.sample_interval = sample_interval

        self._task: asyncio.Task | None = None
        self._running = False
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _run(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.sample_interval)
                self._adjust_once()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.debug(f"Concurrency tuner error: {e}")

    def _adjust_once(self) -> None:
        # Guard against missing attributes
        if not hasattr(self.dispatcher, "max_session_permit"):
            return

        current = int(getattr(self.dispatcher, "max_session_permit", 1))

        # Read metrics
        metrics = self.monitor.metrics
        error_rate = metrics.error_rate
        avg_cpu = metrics.average_cpu_usage or 0.0

        # Heuristic: reduce on high error rate or high CPU; increase on low error and moderate CPU
        target = current
        if error_rate > 0.20 or avg_cpu > 92.0:
            target = max(self.min_concurrency, current - 1)
        elif error_rate < 0.05 and 30.0 < avg_cpu < 85.0:
            target = min(self.max_concurrency, current + 1)

        if target != current:
            try:
                self.dispatcher.max_session_permit = int(target)
                self.logger.info(
                    "Adaptive concurrency update: %s -> %s (cpu=%.1f%%, err=%.2f)",
                    current,
                    target,
                    avg_cpu,
                    error_rate,
                )
            except Exception:
                # Silently ignore if dispatcher is immutable in this version
                pass

        # Track peak observed configured concurrency as a proxy for concurrency peak
        try:
            peak = int(
                getattr(self.monitor.metrics, "concurrent_sessions_peak", 0) or 0
            )
            if current > peak:
                self.monitor.metrics.concurrent_sessions_peak = current
        except Exception:
            pass
