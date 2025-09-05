"""
Lightweight GitHub REST API v3 async client for fetching PR details, reviews,
review comments, and issue comments.

Usage:
    async with GitHubClient(token=os.getenv("GITHUB_TOKEN")) as gh:
        pr = await gh.get_pull_request("owner", "repo", 123)
        reviews = await gh.list_reviews("owner", "repo", 123)
        rcomments = await gh.list_review_comments("owner", "repo", 123)
        icomments = await gh.list_issue_comments("owner", "repo", 123)
"""

from __future__ import annotations

import os
from typing import Any

import aiohttp


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        timeout_s: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.timeout_s = timeout_s
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> GitHubClient:
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": "OptimizedCrawler/1.0",
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        await self._ensure_session()
        assert self._session is not None
        url = f"{self.base_url}{path}"
        async with self._session.get(url, params=params) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(
                    f"GitHub API error {resp.status} for {path}: {text[:200]}"
                )
            return await resp.json()

    async def _paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        per_page: int = 100,
        max_pages: int = 50,
    ) -> list[Any]:
        result: list[Any] = []
        page = 1
        while page <= max_pages:
            q = {"per_page": per_page, "page": page}
            if params:
                q.update(params)
            data = await self._get_json(path, q)
            if not isinstance(data, list):
                # For endpoints that unexpectedly return non-lists
                break
            result.extend(data)
            if len(data) < per_page:
                break
            page += 1
        return result

    async def get_pull_request(
        self, owner: str, repo: str, number: int
    ) -> dict[str, Any]:
        return await self._get_json(f"/repos/{owner}/{repo}/pulls/{number}")

    async def list_reviews(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, Any]]:
        return await self._paginate(f"/repos/{owner}/{repo}/pulls/{number}/reviews")

    async def list_review_comments(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, Any]]:
        return await self._paginate(f"/repos/{owner}/{repo}/pulls/{number}/comments")

    async def list_issue_comments(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, Any]]:
        # PRs are issues under the hood for issue comments
        return await self._paginate(f"/repos/{owner}/{repo}/issues/{number}/comments")

    async def list_pull_files(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, Any]]:
        """List files changed in a pull request."""
        return await self._paginate(f"/repos/{owner}/{repo}/pulls/{number}/files")

    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str
    ) -> tuple[str, str]:
        """Get file content at a specific ref. Returns (encoding, content_str)."""
        data = await self._get_json(
            f"/repos/{owner}/{repo}/contents/{path}", params={"ref": ref}
        )
        import base64

        enc = str(data.get("encoding", ""))
        if enc == "base64":
            raw = base64.b64decode(data.get("content", "") or b"")
            return enc, raw.decode("utf-8", errors="replace")
        return enc, str(data.get("content", "") or "")

    async def get_timeline(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, Any]]:
        """Get PR timeline events for force-push detection. Uses preview header."""
        await self._ensure_session()
        assert self._session is not None

        # Timeline API requires preview header for full access
        headers = {"Accept": "application/vnd.github.mockingbird-preview+json"}
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{number}/timeline"

        async with self._session.get(url, headers=headers) as resp:
            if resp.status >= 400:
                text = await resp.text()
                if resp.status == 404:
                    # Timeline not available, return empty list
                    return []
                raise RuntimeError(f"GitHub API error {resp.status}: {text[:200]}")
            return await resp.json()

    async def post_issue_comment(
        self, owner: str, repo: str, number: int, body: str
    ) -> dict[str, Any]:
        await self._ensure_session()
        assert self._session is not None
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{number}/comments"
        async with self._session.post(url, json={"body": body}) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"GitHub API error {resp.status}: {text[:200]}")
            return await resp.json()
