"""
Async client for HuggingFace Text Embeddings Inference (TEI).

Supports both native TEI `/embed` and OpenAI-compatible `/v1/embeddings` routes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import aiohttp


class TEIEmbeddingsClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        timeout_s: float = 15.0,
        max_retries: int = 1,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max(0, int(max_retries))
        self.headers = headers or {"content-type": "application/json"}

        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> TEIEmbeddingsClient:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout, headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of texts; returns list of embedding vectors (floats)."""
        items = [t if isinstance(t, str) else str(t) for t in texts]
        if not items:
            return []

        # Prefer native TEI /embed; fall back to OpenAI-compatible /v1/embeddings
        try:
            return await self._post_embed(
                {"inputs": items}, "/embed", parse=self._parse_native
            )
        except Exception:
            return await self._post_embed(
                {"input": items, **({"model": self.model} if self.model else {})},
                "/v1/embeddings",
                parse=self._parse_openai,
            )

    async def _post_embed(
        self, payload: dict[str, Any], path: str, *, parse
    ) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                sess = self._session
                if sess is None or sess.closed:
                    timeout = aiohttp.ClientTimeout(total=self.timeout_s)
                    sess = aiohttp.ClientSession(timeout=timeout, headers=self.headers)
                    self._session = sess
                async with sess.post(self.base_url + path, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return parse(data)
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    await asyncio.sleep(min(0.25 * (attempt + 1), 1.0))
                    continue
                break
        raise last_exc if last_exc else RuntimeError("TEI request failed")

    @staticmethod
    def _parse_native(data: Any) -> list[list[float]]:
        # TEI native returns {"data": [[...], [...]]} or just [[...]] depending on version
        if isinstance(data, dict) and "data" in data:
            return list(data["data"])  # type: ignore[return-value]
        if isinstance(data, list):
            return data  # type: ignore[return-value]
        raise ValueError("Unexpected TEI native response format")

    @staticmethod
    def _parse_openai(data: Any) -> list[list[float]]:
        # OpenAI-compatible: {"data": [{"embedding": [...]} ...]}
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            out: list[list[float]] = []
            for item in data["data"]:
                emb = item.get("embedding") if isinstance(item, dict) else None
                if isinstance(emb, list):
                    out.append(emb)
            if out:
                return out
        raise ValueError("Unexpected OpenAI embeddings response format")
