"""
Minimal async Qdrant HTTP client for collection management, upserts, and search.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import aiohttp


class QdrantClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 15.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.headers = headers or {"content-type": "application/json"}
        if api_key:
            self.headers["api-key"] = api_key
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> QdrantClient:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout, headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def ensure_collection(
        self,
        name: str,
        *,
        size: int,
        distance: str = "Cosine",
        vectors_name: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        sess = await self._get_session()
        async with sess.get(f"{self.base_url}/collections/{name}") as r:
            if r.status == 200:
                return
            if r.status not in (404,):
                r.raise_for_status()

        body: dict[str, Any]
        if vectors_name:
            body = {"vectors": {vectors_name: {"size": size, "distance": distance}}}
        else:
            body = {"vectors": {"size": size, "distance": distance}}

        sess = await self._get_session(timeout_s)
        async with sess.put(f"{self.base_url}/collections/{name}", json=body) as r:
            r.raise_for_status()

    async def upsert(
        self,
        name: str,
        points: Iterable[dict[str, Any]],
        *,
        wait: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        payload = {"points": list(points)}
        params = {"wait": "true" if wait else "false"}
        sess = await self._get_session(timeout_s)
        async with sess.put(
            f"{self.base_url}/collections/{name}/points",
            params=params,
            json=payload,
        ) as r:
            r.raise_for_status()
            return await r.json()

    async def get_collection(self, name: str) -> dict[str, Any]:
        sess = await self._get_session()
        async with sess.get(f"{self.base_url}/collections/{name}") as r:
            r.raise_for_status()
            return await r.json()

    async def count_points(
        self,
        name: str,
        *,
        exact: bool = True,
        timeout_s: float | None = None,
    ) -> int:
        body: dict[str, Any] = {"exact": bool(exact)}
        sess = await self._get_session(timeout_s)
        async with sess.post(
            f"{self.base_url}/collections/{name}/points/count",
            json=body,
        ) as r:
            r.raise_for_status()
            data = await r.json()
            try:
                return int(data.get("result", {}).get("count", 0))
            except Exception:
                return 0

    async def scroll_points(
        self,
        name: str,
        *,
        limit: int = 1,
        with_vectors: bool = False,
        with_payload: bool = True,
        query_filter: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "limit": int(limit),
            "with_vectors": bool(with_vectors),
            "with_payload": bool(with_payload),
        }
        if query_filter:
            body["filter"] = query_filter
        sess = await self._get_session(timeout_s)
        async with sess.post(
            f"{self.base_url}/collections/{name}/points/scroll",
            json=body,
        ) as r:
            r.raise_for_status()
            return await r.json()

    async def search(
        self,
        name: str,
        *,
        vector: list[float],
        limit: int = 5,
        with_payload: bool = True,
        with_vectors: bool = False,
        query_filter: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "vector": vector,
            "limit": int(limit),
            "with_payload": bool(with_payload),
            "with_vectors": bool(with_vectors),
        }
        if query_filter:
            body["filter"] = query_filter
        sess = await self._get_session(timeout_s)
        async with sess.post(
            f"{self.base_url}/collections/{name}/points/search",
            json=body,
        ) as r:
            r.raise_for_status()
            return await r.json()

    async def set_payload(
        self,
        name: str,
        *,
        payload: dict[str, Any],
        ids: list[str] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Set payload values for points by IDs."""
        body: dict[str, Any] = {"payload": payload}
        if ids:
            body["points"] = ids
        sess = await self._get_session(timeout_s)
        async with sess.put(
            f"{self.base_url}/collections/{name}/points/payload",
            json=body,
        ) as r:
            r.raise_for_status()
            return await r.json()

    async def _get_session(
        self, timeout_s: float | None = None
    ) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=(timeout_s or self.timeout_s))
            self._session = aiohttp.ClientSession(timeout=timeout, headers=self.headers)
        return self._session
