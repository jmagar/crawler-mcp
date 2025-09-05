"""Progress tracking middleware for long-running MCP operations."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request, Response


@dataclass
class ProgressTracker:
    operation_id: str
    start_time: float = field(default_factory=time.time)
    progress: int = 0
    total: int = 100
    status: str = "in_progress"

    def update(
        self,
        progress: int | None = None,
        total: int | None = None,
        status: str | None = None,
    ) -> None:
        if progress is not None:
            self.progress = progress
        if total is not None:
            self.total = total
        if status is not None:
            self.status = status

    def progress_percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return min(100.0, max(0.0, (self.progress / self.total) * 100))

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def estimated_time_remaining(self) -> float | None:
        percent = self.progress_percentage()
        if percent <= 0:
            return None
        elapsed = self.elapsed_time()
        return max(0.0, (elapsed * (100 - percent)) / percent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "progress": self.progress,
            "total": self.total,
            "status": self.status,
            "progress_percentage": self.progress_percentage(),
            "elapsed_time": self.elapsed_time(),
            "estimated_time_remaining": self.estimated_time_remaining(),
        }


class ProgressMiddleware:
    """Middleware to track and manage progress for long-running operations."""

    def __init__(self, app: Callable[[Request], Awaitable[Response]]):
        self.app = app
        self._active: dict[str, ProgressTracker] = {}

    async def __call__(
        self, request: Request
    ) -> Response:  # pragma: no cover - used by transport
        return await self.app(request)

    def create_tracker(self, operation_id: str) -> ProgressTracker:
        tracker = ProgressTracker(operation_id)
        self._active[operation_id] = tracker
        return tracker

    def get_tracker(self, operation_id: str) -> ProgressTracker | None:
        return self._active.get(operation_id)

    def remove_tracker(self, operation_id: str) -> ProgressTracker | None:
        return self._active.pop(operation_id, None)

    def list_active_operations(self) -> dict[str, dict[str, Any]]:
        return {op_id: t.to_dict() for op_id, t in self._active.items()}


async def _dummy_app(_request: Request) -> Response:  # pragma: no cover
    return Response("Middleware not initialized", status_code=500)


progress_middleware = ProgressMiddleware(_dummy_app)
