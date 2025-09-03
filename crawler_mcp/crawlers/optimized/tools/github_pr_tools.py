from __future__ import annotations

import contextlib
import fnmatch
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastmcp import Context, FastMCP

from ..clients.github_client import GitHubClient
from ..clients.qdrant_http_client import QdrantClient
from ..utils.github_suggestions import (
    apply_suggestion as _apply_sugg,
)
from ..utils.github_suggestions import (
    extract_suggestions,
)
from ..utils.github_suggestions import (
    unified_diff as _udiff,
)


def _to_iso(ts: str | None) -> str:
    try:
        if not ts:
            return ""
        # ensure ISO string
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat()
    except Exception:
        return str(ts or "")


async def list_pr_items_impl(
    owner: str, repo: str, pr_number: int
) -> list[dict[str, Any]]:
    token = os.getenv("GITHUB_TOKEN", "")
    async with GitHubClient(token=token, timeout_s=30.0) as gh:
        pr = await gh.get_pull_request(owner, repo, pr_number)
        reviews = await gh.list_reviews(owner, repo, pr_number)
        rcomments = await gh.list_review_comments(owner, repo, pr_number)
        icomments = await gh.list_issue_comments(owner, repo, pr_number)

    base_items: list[dict[str, Any]] = []
    # Overview
    base_items.append(
        {
            "item_type": "pr_overview",
            "owner": owner,
            "repo": repo,
            "pr_number": pr_number,
            "author": (pr.get("user", {}) or {}).get("login", ""),
            "created_at": _to_iso(pr.get("created_at")),
            "updated_at": _to_iso(pr.get("updated_at")),
            "canonical_url": pr.get("html_url", ""),
            "item_id": pr.get("id"),
            "title": pr.get("title", ""),
            "pr_state": pr.get("state", ""),
            "pr_merged": bool(pr.get("merged", False)),
            "body": pr.get("body", "") or "",
        }
    )

    # Reviews
    for rv in reviews:
        base_items.append(
            {
                "item_type": "pr_review",
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "author": (rv.get("user", {}) or {}).get("login", ""),
                "created_at": _to_iso(rv.get("submitted_at")),
                "updated_at": _to_iso(rv.get("submitted_at")),
                "canonical_url": rv.get("html_url")
                or f"https://github.com/{owner}/{repo}/pull/{pr_number}#pullrequestreview-{rv.get('id')}",
                "item_id": rv.get("id"),
                "review_state": rv.get("state", ""),
                "body": rv.get("body", "") or "",
            }
        )

    # Review comments
    for c in rcomments:
        is_outdated = c.get("position") is None
        base_items.append(
            {
                "item_type": "pr_review_comment",
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "author": (c.get("user", {}) or {}).get("login", ""),
                "created_at": _to_iso(c.get("created_at")),
                "updated_at": _to_iso(c.get("updated_at")),
                "canonical_url": c.get("html_url")
                or f"https://github.com/{owner}/{repo}/pull/{pr_number}#discussion_r{c.get('id')}",
                "item_id": c.get("id"),
                "path": c.get("path", ""),
                "line": c.get("line") or c.get("original_line"),
                "original_line": c.get("original_line"),
                "position": c.get("position"),
                "original_position": c.get("original_position"),
                "commit_id": c.get("commit_id"),
                "original_commit_id": c.get("original_commit_id"),
                "diff_hunk": c.get("diff_hunk", ""),
                "in_reply_to_id": c.get("in_reply_to_id"),
                "pull_request_review_id": c.get("pull_request_review_id"),
                "is_outdated": bool(is_outdated),
                "is_resolved": False,
                "body": c.get("body", "") or "",
                # Suggestion parsing
                "suggestions": [
                    s.content for s in extract_suggestions(c.get("body", "") or "")
                ],
            }
        )

    # Conversation comments
    for c in icomments:
        base_items.append(
            {
                "item_type": "pr_conversation_comment",
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "author": (c.get("user", {}) or {}).get("login", ""),
                "created_at": _to_iso(c.get("created_at")),
                "updated_at": _to_iso(c.get("updated_at")),
                "canonical_url": c.get("html_url")
                or f"https://github.com/{owner}/{repo}/pull/{pr_number}",
                "item_id": c.get("id"),
                "body": c.get("body", "") or "",
            }
        )

    return base_items


def _apply_filters(
    items: list[dict[str, Any]], filters: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not filters:
        return items
    out = []
    authors = {a.lower() for a in (filters.get("authors") or [])}
    bots = {b.lower() for b in (filters.get("bots") or [])}
    item_types = set(filters.get("item_types") or [])
    only_unresolved = bool(filters.get("only_unresolved") or False)
    min_length = int(filters.get("min_length") or 0)
    after = str(filters.get("after") or "").strip() or None
    before = str(filters.get("before") or "").strip() or None
    globs = list(filters.get("file_globs") or [])
    for it in items:
        a = str(it.get("author", "")).lower()
        if authors and a not in authors:
            continue
        if bots and a in bots:
            continue
        if item_types and it.get("item_type") not in item_types:
            continue
        if only_unresolved and bool(it.get("is_resolved", False)) is True:
            continue
        if min_length and len(str(it.get("body", ""))) < min_length:
            continue
        if globs:
            p = str(it.get("path", ""))
            if p and not any(fnmatch.fnmatch(p, g) for g in globs):
                continue
        if after or before:
            ts = it.get("created_at") or ""
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if after and dt < datetime.fromisoformat(after.replace("Z", "+00:00")):
                    continue
                if before and dt > datetime.fromisoformat(
                    before.replace("Z", "+00:00")
                ):
                    continue
            except Exception:
                pass
        out.append(it)
    return out


def register_github_pr_tools(mcp: FastMCP) -> None:
    # Local registry file for resolved tracking
    REGISTRY = Path(".pr_resolved_registry.json")

    def _load_registry() -> dict[str, Any]:
        try:
            if REGISTRY.exists():
                return json.loads(REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_registry(data: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            REGISTRY.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    @mcp.tool
    async def list_pr_items(
        ctx: Context,
        owner: str,
        repo: str,
        pr_number: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """List normalized PR items (overview, reviews, review_comments, conversation_comments).

        Filters:
          - authors: ["alice", "bob"]
          - bots: ["coderabbitai[bot]", "copilot-pull-request-reviewer[bot]"]
          - item_types: ["pr_review_comment"]
          - file_globs: ["**/*.py"]
          - only_unresolved: true
          - min_length: 20
          - after/before: ISO timestamps
        """
        items = await list_pr_items_impl(owner, repo, pr_number)
        # Merge local resolved registry flags
        reg = _load_registry().get(f"{owner}/{repo}#{pr_number}", {})
        for it in items:
            iid = str(it.get("item_id") or it.get("canonical_url") or "")
            if iid and iid in reg:
                it["is_resolved"] = bool(reg.get(iid))
        return _apply_filters(items, filters)

    @mcp.tool
    async def get_file_context(
        ctx: Context,
        owner: str,
        repo: str,
        ref: str,
        path: str,
        start_line: int,
        end_line: int,
        commit_id: str | None = None,
        original_commit_id: str | None = None,
    ) -> dict[str, Any]:
        """Return file snippet from GitHub at ref (inclusive lines). If commit ids provided,
        also includes head/base snippets for review context."""
        token = os.getenv("GITHUB_TOKEN", "")
        async with GitHubClient(token=token, timeout_s=30.0) as gh:
            _, content = await gh.get_file_content(owner, repo, path, ref)
            head_snip = base_snip = ""
            s = max(1, int(start_line))
            e = max(s, int(end_line))
            if commit_id:
                try:
                    _, head_text = await gh.get_file_content(
                        owner, repo, path, commit_id
                    )
                    head_snip = "\n".join(head_text.splitlines()[s - 1 : e])
                except Exception:
                    head_snip = ""
            if original_commit_id:
                try:
                    _, base_text = await gh.get_file_content(
                        owner, repo, path, original_commit_id
                    )
                    base_snip = "\n".join(base_text.splitlines()[s - 1 : e])
                except Exception:
                    base_snip = ""
        lines = content.splitlines()
        snippet = "\n".join(lines[s - 1 : e])
        out: dict[str, Any] = {
            "path": path,
            "ref": ref,
            "start": s,
            "end": e,
            "snippet": snippet,
        }
        if head_snip or base_snip:
            out.update(
                {
                    "commit_id": commit_id,
                    "original_commit_id": original_commit_id,
                    "head_snippet": head_snip,
                    "base_snippet": base_snip,
                }
            )
        return out

    @mcp.tool
    async def apply_suggestions(
        ctx: Context,
        owner: str,
        repo: str,
        pr_number: int,
        strategy: str = "dry-run",
        include_bots: bool = True,
        include_humans: bool = True,
        only_unresolved: bool = False,
    ) -> dict[str, Any]:
        """Parse and apply GitHub ```suggestion blocks from review comments.

        strategy: "dry-run" returns diffs; "commit" writes files and commits locally.
        Requires local working copy of the repository for commit mode.
        """
        items = await list_pr_items_impl(owner, repo, pr_number)
        reg = _load_registry().get(f"{owner}/{repo}#{pr_number}", {})
        changes: list[dict[str, Any]] = []
        for it in items:
            if it.get("item_type") != "pr_review_comment":
                continue
            author = str(it.get("author", "")).lower()
            is_bot = (
                author.endswith("[bot]")
                or ("copilot" in author)
                or ("coderabbit" in author)
            )
            if (not include_bots and is_bot) or (not include_humans and not is_bot):
                continue
            if only_unresolved and bool(
                it.get("is_resolved", False) or reg.get(str(it.get("item_id")), False)
            ):
                continue
            path = str(it.get("path", ""))
            if not path:
                continue
            suggestions = it.get("suggestions") or []
            if not suggestions:
                continue
            start = int(it.get("start_line") or it.get("line") or 1)
            end = int(it.get("end_line") or it.get("line") or start)
            try:
                with open(path, encoding="utf-8") as f:
                    before = f.read()
            except Exception:
                continue
            after = before
            for s in suggestions:
                after = _apply_sugg(
                    after, start_line=start, end_line=end, suggestion_text=str(s)
                )
            if after != before:
                diff = _udiff(path, before, after)
                changes.append({"path": path, "diff": diff})
                if strategy == "commit":
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(after)
                    except Exception:
                        pass
        if strategy == "commit" and changes:
            import subprocess as sp

            try:
                sp.run(["git", "add", "-A"], check=False)
                sp.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        f"chore: apply suggestions for PR #{pr_number}",
                    ],
                    check=False,
                )
            except Exception:
                pass
        return {"applied": strategy == "commit", "changes": changes}

    @mcp.tool
    async def mark_items_resolved(
        ctx: Context,
        owner: str,
        repo: str,
        pr_number: int,
        resolved: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark items as resolved/unchecked in local registry and attempt to update Qdrant payload.

        Filters: authors, bots, item_types, file_globs, min_length, after/before.
        """
        items = await list_pr_items_impl(owner, repo, pr_number)
        items = _apply_filters(items, filters)
        reg_all = _load_registry()
        key = f"{owner}/{repo}#{pr_number}"
        reg = reg_all.get(key, {})
        updated = 0
        for it in items:
            iid = str(it.get("item_id") or it.get("canonical_url") or "")
            if not iid:
                continue
            reg[iid] = bool(resolved)
            updated += 1
        reg_all[key] = reg
        _save_registry(reg_all)

        # Best-effort Qdrant payload update (requires env vars)
        try:
            qurl = os.getenv("OPTIMIZED_CRAWLER_QDRANT_URL") or os.getenv("QDRANT_URL")
            qcol = os.getenv("OPTIMIZED_CRAWLER_QDRANT_COLLECTION") or os.getenv(
                "QDRANT_COLLECTION"
            )
            qkey = os.getenv("OPTIMIZED_CRAWLER_QDRANT_API_KEY") or os.getenv(
                "QDRANT_API_KEY"
            )
            if qurl and qcol:
                async with QdrantClient(qurl, api_key=qkey, timeout_s=10.0) as qc:
                    # Construct filter for owner/repo/pr and optional item_types
                    must = [
                        {"key": "owner", "match": {"value": owner}},
                        {"key": "repo", "match": {"value": repo}},
                        {"key": "pr_number", "match": {"value": int(pr_number)}},
                    ]
                    if filters and filters.get("item_types"):
                        # build should for any item_types
                        should = [
                            {"key": "item_type", "match": {"value": t}}
                            for t in (filters.get("item_types") or [])
                        ]
                        qf = {"must": must, "should": should, "minimum_should_match": 1}
                    else:
                        qf = {"must": must}
                    data = await qc.scroll_points(
                        qcol,
                        limit=10000,
                        with_vectors=False,
                        with_payload=False,
                        query_filter=qf,
                    )
                    pts = (
                        (data.get("result", {}) or {}).get("points", [])
                        if isinstance(data, dict)
                        else []
                    )
                    ids = [p.get("id") for p in pts if p.get("id") is not None]
                    if ids:
                        await qc.set_payload(
                            qcol, payload={"resolved": bool(resolved)}, ids=ids
                        )
        except Exception:
            pass

        return {"updated": updated, "resolved": bool(resolved)}

    @mcp.tool
    async def create_branch_from_pr(
        ctx: Context, owner: str, repo: str, pr_number: int, branch_name: str
    ) -> dict[str, Any]:
        """Create and switch to a local branch for fixes. Assumes repo is checked out locally."""
        import subprocess as sp

        res = sp.run(
            ["git", "switch", "-c", branch_name], capture_output=True, text=True
        )
        ok = res.returncode == 0
        return {"ok": ok, "stdout": res.stdout, "stderr": res.stderr}

    @mcp.tool
    async def post_comment(
        ctx: Context, owner: str, repo: str, pr_number: int, body: str
    ) -> dict[str, Any]:
        """Post a comment on the PR conversation (Issues API)."""
        token = os.getenv("GITHUB_TOKEN", "")
        async with GitHubClient(token=token, timeout_s=30.0) as gh:
            out = await gh.post_issue_comment(owner, repo, pr_number, body)
        return {"ok": True, "id": out.get("id"), "url": out.get("html_url")}
