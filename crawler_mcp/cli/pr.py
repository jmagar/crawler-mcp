"""
CLI for GitHub PR operations tailored for LLM tooling workflows.

Usage examples:
  # List items (JSON/NDJSON) - automatically saved to structured paths
  uv run python -m crawler_mcp.cli.pr pr-list \
    --owner OWNER --repo REPO --pr 123 --item-types pr_review_comment --only-unresolved \
    --ndjson

  # Get file context
  uv run python -m crawler_mcp.cli.pr pr-context \
    --owner OWNER --repo REPO --ref main --path path/to/file.py --start 10 --end 40

  # Apply suggestions (dry-run/commit)
  uv run python -m crawler_mcp.cli.pr pr-apply-suggestions \
    --owner OWNER --repo REPO --pr 123 --strategy dry-run --only-unresolved

  # Create fix branch
  uv run python -m crawler_mcp.cli.pr pr-branch \
    --owner OWNER --repo REPO --pr 123 --branch pr-123-fixes

  # Post a comment
  uv run python -m crawler_mcp.cli.pr pr-comment \
    --owner OWNER --repo REPO --pr 123 --body "Applied suggestions."

  # Mark items resolved
  uv run python -m crawler_mcp.cli.pr pr-mark-resolved \
    --owner OWNER --repo REPO --pr 123 --resolved true --item-types pr_review_comment
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from crawler_mcp.clients.github_client import GitHubClient
from crawler_mcp.clients.qdrant_http_client import QdrantClient
from crawler_mcp.tools.github_pr_tools import _apply_filters, list_pr_items_impl
from crawler_mcp.utils.github_suggestions import (
    apply_suggestion as _apply_sugg,
)
from crawler_mcp.utils.github_suggestions import (
    unified_diff as _udiff,
)
from crawler_mcp.utils.output_manager import OutputManager


def _get_output_manager() -> OutputManager:
    # Ensure env is loaded for OptimizedConfig.from_env() and clients
    pkg_root = Path(__file__).resolve().parents[1]
    load_dotenv(pkg_root / ".env", override=False)
    load_dotenv(pkg_root.parent / ".env", override=False)
    load_dotenv(pkg_root / "crawlers" / "optimized" / ".env", override=False)
    """Get OutputManager instance with default settings."""
    return OutputManager()


def _load_registry() -> dict[str, Any]:
    try:
        registry_path = _get_output_manager().get_pr_registry_path()
        if registry_path.exists():
            return json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_registry(data: dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        registry_path = _get_output_manager().get_pr_registry_path()
        # Ensure parent directory exists
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _build_filters_from_args(ns: argparse.Namespace) -> dict[str, Any]:
    f: dict[str, Any] = {}

    # Parse enhanced filters from JSON if provided
    if hasattr(ns, "filters") and ns.filters:
        with contextlib.suppress(json.JSONDecodeError):
            f.update(json.loads(ns.filters))

    # Override with individual arguments (these take precedence)
    if ns.authors:
        f["authors"] = [s.strip() for s in ns.authors.split(",") if s.strip()]
    if ns.bots:
        f["bots"] = [s.strip() for s in ns.bots.split(",") if s.strip()]
    if ns.item_types:
        f["item_types"] = [s.strip() for s in ns.item_types.split(",") if s.strip()]
    if ns.file_globs:
        f["file_globs"] = [s.strip() for s in ns.file_globs.split(",") if s.strip()]
    if ns.only_unresolved:
        f["only_unresolved"] = True
    if ns.min_length is not None:
        f["min_length"] = int(ns.min_length)
    if ns.after:
        f["after"] = str(ns.after)
    if ns.before:
        f["before"] = str(ns.before)
    return f


def cmd_pr_list(ns: argparse.Namespace) -> int:
    items = ns.loop.run_until_complete(list_pr_items_impl(ns.owner, ns.repo, ns.pr))
    # merge resolved flags
    reg = _load_registry().get(f"{ns.owner}/{ns.repo}#{ns.pr}", {})
    for it in items:
        iid = str(it.get("item_id") or it.get("canonical_url") or "")
        if iid and iid in reg:
            it["is_resolved"] = bool(reg.get(iid))

    # Apply enhanced filtering
    filters = _build_filters_from_args(ns)
    filtered_items, filtered_count = _apply_filters(items, filters)
    items = filtered_items

    # Always save to structured output path
    output_mgr = _get_output_manager()
    paths = output_mgr.get_pr_output_paths(ns.owner, ns.repo, ns.pr)

    if ns.ndjson:
        # Save as NDJSON to structured location
        with open(paths["items"], "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"✅ Items saved to: {paths['items']}")
    else:
        # Print to stdout and optionally save as JSON
        print(json.dumps(items, ensure_ascii=False, indent=2))
        if len(items) > 0:  # Only save if there are items
            json_path = paths["items"].with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"✅ Items also saved to: {json_path}")

    return 0


def cmd_pr_context(ns: argparse.Namespace) -> int:
    token = os.getenv("GITHUB_TOKEN", "")

    async def _go():
        async with GitHubClient(token=token, timeout_s=20.0) as gh:
            _, content = await gh.get_file_content(ns.owner, ns.repo, ns.path, ns.ref)
            s = max(1, int(ns.start))
            e = max(s, int(ns.end))
            lines = content.splitlines()
            out = {
                "path": ns.path,
                "ref": ns.ref,
                "start": s,
                "end": e,
                "snippet": "\n".join(lines[s - 1 : e]),
            }
            if ns.commit_id or ns.original_commit_id:
                head_snip = base_snip = ""
                if ns.commit_id:
                    try:
                        _, head_text = await gh.get_file_content(
                            ns.owner, ns.repo, ns.path, ns.commit_id
                        )
                        head_snip = "\n".join(head_text.splitlines()[s - 1 : e])
                    except Exception:
                        pass
                if ns.original_commit_id:
                    try:
                        _, base_text = await gh.get_file_content(
                            ns.owner, ns.repo, ns.path, ns.original_commit_id
                        )
                        base_snip = "\n".join(base_text.splitlines()[s - 1 : e])
                    except Exception:
                        pass
                out["head_snippet"] = head_snip
                out["base_snippet"] = base_snip
            print(json.dumps(out, ensure_ascii=False, indent=2))

    ns.loop.run_until_complete(_go())
    return 0


def cmd_pr_apply_suggestions(ns: argparse.Namespace) -> int:
    items = ns.loop.run_until_complete(list_pr_items_impl(ns.owner, ns.repo, ns.pr))
    reg = _load_registry().get(f"{ns.owner}/{ns.repo}#{ns.pr}", {})
    changes: list[dict[str, Any]] = []
    for it in items:
        if it.get("item_type") != "pr_review_comment":
            continue
        a = str(it.get("author", "")).lower()
        is_bot = a.endswith("[bot]") or ("copilot" in a) or ("coderabbit" in a)
        if (not ns.include_bots and is_bot) or (not ns.include_humans and not is_bot):
            continue
        if ns.only_unresolved and bool(
            it.get("is_resolved", False) or reg.get(str(it.get("item_id")), False)
        ):
            continue
        path = str(it.get("path", ""))
        if not path:
            continue
        suggs = it.get("suggestions") or []
        if not suggs:
            continue
        start = int(it.get("start_line") or it.get("line") or 1)
        end = int(it.get("end_line") or it.get("line") or start)
        try:
            with open(path, encoding="utf-8") as f:
                before = f.read()
        except Exception:
            continue
        after = before
        for s in suggs:
            after = _apply_sugg(
                after, start_line=start, end_line=end, suggestion_text=str(s)
            )
        if after != before:
            diff = _udiff(path, before, after)
            changes.append({"path": path, "diff": diff})
            if ns.strategy == "commit":
                try:
                    with open(path, "w", encoding="utf-8") as wf:
                        wf.write(after)
                except Exception:
                    pass
    if ns.strategy == "commit" and changes:
        try:
            import subprocess as sp

            sp.run(["git", "add", "-A"], check=False)
            sp.run(
                ["git", "commit", "-m", f"chore: apply suggestions for PR #{ns.pr}"],
                check=False,
            )
        except Exception:
            pass
    result = {"applied": ns.strategy == "commit", "changes": changes}

    # Print to stdout
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Also save to structured path
    output_mgr = _get_output_manager()
    paths = output_mgr.get_pr_output_paths(ns.owner, ns.repo, ns.pr)

    with open(paths["suggestions"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ Suggestions result saved to: {paths['suggestions']}")

    return 0


def cmd_pr_branch(ns: argparse.Namespace) -> int:
    import subprocess as sp

    res = sp.run(["git", "switch", "-c", ns.branch], capture_output=True, text=True)
    ok = res.returncode == 0
    print(
        json.dumps(
            {"ok": ok, "stdout": res.stdout, "stderr": res.stderr}, ensure_ascii=False
        )
    )
    return 0 if ok else 1


def cmd_pr_comment(ns: argparse.Namespace) -> int:
    token = os.getenv("GITHUB_TOKEN", "")

    async def _go():
        async with GitHubClient(token=token, timeout_s=20.0) as gh:
            out = await gh.post_issue_comment(ns.owner, ns.repo, ns.pr, ns.body)
        print(
            json.dumps(
                {"ok": True, "id": out.get("id"), "url": out.get("html_url")},
                ensure_ascii=False,
            )
        )

    ns.loop.run_until_complete(_go())
    return 0


def cmd_pr_mark_resolved(ns: argparse.Namespace) -> int:
    items = ns.loop.run_until_complete(list_pr_items_impl(ns.owner, ns.repo, ns.pr))
    filters = _build_filters_from_args(ns)
    filtered_items, filtered_count = _apply_filters(items, filters)
    items = filtered_items
    reg_all = _load_registry()
    key = f"{ns.owner}/{ns.repo}#{ns.pr}"
    reg = reg_all.get(key, {})
    for it in items:
        iid = str(it.get("item_id") or it.get("canonical_url") or "")
        if not iid:
            continue
        reg[iid] = bool(ns.resolved)
    reg_all[key] = reg
    _save_registry(reg_all)
    # Best-effort Qdrant payload update
    try:
        qurl = os.getenv("OPTIMIZED_CRAWLER_QDRANT_URL") or os.getenv("QDRANT_URL")
        qcol = os.getenv("OPTIMIZED_CRAWLER_QDRANT_COLLECTION") or os.getenv(
            "QDRANT_COLLECTION"
        )
        qkey = os.getenv("OPTIMIZED_CRAWLER_QDRANT_API_KEY") or os.getenv(
            "QDRANT_API_KEY"
        )
        if qurl and qcol:

            async def _upd():
                async with QdrantClient(qurl, api_key=qkey, timeout_s=10.0) as qc:
                    must = [
                        {"key": "owner", "match": {"value": ns.owner}},
                        {"key": "repo", "match": {"value": ns.repo}},
                        {"key": "pr_number", "match": {"value": int(ns.pr)}},
                    ]
                    if ns.item_types:
                        should = [
                            {"key": "item_type", "match": {"value": t}}
                            for t in ns.item_types.split(",")
                            if t
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
                            qcol, payload={"resolved": bool(ns.resolved)}, ids=ids
                        )

            ns.loop.run_until_complete(_upd())
    except Exception:
        pass
    result = {"updated": len(items), "resolved": bool(ns.resolved)}

    # Print to stdout
    print(json.dumps(result, ensure_ascii=False))

    # Also save resolved status to structured path
    output_mgr = _get_output_manager()
    paths = output_mgr.get_pr_output_paths(ns.owner, ns.repo, ns.pr)

    with open(paths["resolved"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ Resolution status saved to: {paths['resolved']}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("pr-cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common opts
    def _add_common(s):
        s.add_argument("--owner", required=True)
        s.add_argument("--repo", required=True)
        s.add_argument("--pr", type=int, required=True)

    # list
    s = sub.add_parser("pr-list")
    _add_common(s)
    s.add_argument("--authors", default=None)
    s.add_argument("--bots", default=None)
    s.add_argument("--item-types", default=None)
    s.add_argument("--file-globs", default=None)
    s.add_argument("--only-unresolved", action="store_true")
    s.add_argument("--min-length", type=int, default=None)
    s.add_argument("--after", default=None)
    s.add_argument("--before", default=None)
    s.add_argument(
        "--ndjson", action="store_true", help="Save as NDJSON format instead of JSON"
    )
    s.add_argument(
        "--filters", default=None, help="JSON string with enhanced filtering options"
    )

    # context
    s = sub.add_parser("pr-context")
    s.add_argument("--owner", required=True)
    s.add_argument("--repo", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--path", required=True)
    s.add_argument("--start", type=int, required=True)
    s.add_argument("--end", type=int, required=True)
    s.add_argument("--commit-id", default=None)
    s.add_argument("--original-commit-id", default=None)

    # apply suggestions
    s = sub.add_parser("pr-apply-suggestions")
    _add_common(s)
    s.add_argument("--strategy", default="dry-run", choices=["dry-run", "commit"])
    s.add_argument(
        "--include-bots",
        type=lambda x: x.lower() in {"1", "true", "yes", "on"},
        default=True,
    )
    s.add_argument(
        "--include-humans",
        type=lambda x: x.lower() in {"1", "true", "yes", "on"},
        default=True,
    )
    s.add_argument("--only-unresolved", action="store_true")

    # branch
    s = sub.add_parser("pr-branch")
    _add_common(s)
    s.add_argument("--branch", required=True)

    # comment
    s = sub.add_parser("pr-comment")
    _add_common(s)
    s.add_argument("--body", required=True)

    # mark-resolved
    s = sub.add_parser("pr-mark-resolved")
    _add_common(s)
    s.add_argument(
        "--resolved",
        type=lambda x: x.lower() in {"1", "true", "yes", "on"},
        default=True,
    )
    s.add_argument("--authors", default=None)
    s.add_argument("--bots", default=None)
    s.add_argument("--item-types", default=None)
    s.add_argument("--file-globs", default=None)
    s.add_argument("--only-unresolved", action="store_true")
    s.add_argument("--min-length", type=int, default=None)
    s.add_argument("--after", default=None)
    s.add_argument("--before", default=None)

    return p


def main() -> None:
    p = build_parser()
    ns = p.parse_args()
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        ns.loop = loop
        if ns.cmd == "pr-list":
            raise SystemExit(cmd_pr_list(ns))
        if ns.cmd == "pr-context":
            raise SystemExit(cmd_pr_context(ns))
        if ns.cmd == "pr-apply-suggestions":
            raise SystemExit(cmd_pr_apply_suggestions(ns))
        if ns.cmd == "pr-branch":
            raise SystemExit(cmd_pr_branch(ns))
        if ns.cmd == "pr-comment":
            raise SystemExit(cmd_pr_comment(ns))
        if ns.cmd == "pr-mark-resolved":
            raise SystemExit(cmd_pr_mark_resolved(ns))
        raise SystemExit(2)
    finally:
        with contextlib.suppress(Exception):
            loop.close()


if __name__ == "__main__":
    main()
