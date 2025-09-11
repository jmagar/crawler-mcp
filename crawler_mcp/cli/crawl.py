"""
CLI runner for the optimized crawler.

Usage:
  uv run python -m crawler_mcp.cli.crawl \
      --url https://gofastmcp.com \
      --output-html crawl_all.html \
      --output-ndjson pages.ndjson \
      --concurrency 16 \
      --max-urls 1000 \
      --content-validation false \
      --doc-relax "/python-sdk/" --doc-relax "/servers/" --doc-relax "/integrations/"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

from crawler_mcp.core import CrawlOrchestrator
from crawler_mcp.utils.log_manager import LogManager
from crawler_mcp.utils.output_manager import OutputManager


def _bool(x: str) -> bool:
    return x.lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    # Ensure env is loaded for settings
    pkg_root = Path(__file__).resolve().parents[1]
    if load_dotenv:
        load_dotenv(pkg_root / ".env", override=False)
        load_dotenv(pkg_root.parent / ".env", override=False)
    p = argparse.ArgumentParser("optimized-crawler")
    p.add_argument(
        "--url",
        required=False,
        default="",
        help="Start URL to crawl (required unless using --qdrant-search)",
    )
    p.add_argument("--max-urls", type=int, default=1000, help="Max URLs to crawl")
    p.add_argument("--concurrency", type=int, default=16, help="Max concurrent crawls")
    p.add_argument(
        "--content-validation",
        type=_bool,
        default=False,
        help="Enable content quality validation (default: false)",
    )
    p.add_argument(
        "--doc-relax",
        action="append",
        default=[],
        help="Regex pattern to relax validation for doc-like URLs (can repeat)",
    )
    p.add_argument(
        "--output-dir",
        default="./.crawl4ai",
        help="Base output directory (default: ./.crawl4ai)",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup rotation (overwrite previous backup)",
    )
    p.add_argument(
        "--clean-outputs",
        action="store_true",
        help="Clean all outputs before starting crawl",
    )
    p.add_argument(
        "--skip-output",
        action="store_true",
        help="Don't save outputs to disk (useful for testing)",
    )
    p.add_argument(
        "--page-timeout-ms",
        type=int,
        default=30000,
        help="Per-page timeout in milliseconds (default: 30000)",
    )
    p.add_argument(
        "--aggressive",
        action="store_true",
        help="Enable aggressive tuning (higher concurrency, shorter timeouts)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Crawl4AI cache",
    )
    p.add_argument(
        "--javascript",
        type=_bool,
        default=None,
        help="Enable JavaScript during crawl (default: use config)",
    )
    p.add_argument(
        "--no-js-retry",
        action="store_true",
        help="Disable JS retry pass for failed URLs",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="(Deprecated) Stream results; use ENABLE_STREAMING=1 env instead",
    )
    p.add_argument(
        "--preset-13700k",
        action="store_true",
        help="Apply performance preset for i7-13700K + 32GB RAM",
    )
    # Embeddings
    p.add_argument(
        "--embeddings",
        type=_bool,
        default=None,
        help="Enable TEI embeddings for each page content",
    )
    p.add_argument(
        "--tei-endpoint",
        default=None,
        help="TEI base URL (e.g., http://steamy-wsl:8080)",
    )
    p.add_argument(
        "--tei-batch",
        type=int,
        default=None,
        help="TEI batch size (default 16)",
    )
    p.add_argument(
        "--tei-parallel",
        type=int,
        default=None,
        help="Concurrent TEI batch POSTs (default 4)",
    )
    p.add_argument(
        "--tei-timeout-s", type=float, default=None, help="TEI request timeout seconds"
    )
    p.add_argument(
        "--tei-max-concurrent",
        type=int,
        default=None,
        help="Cap client parallel by TEI server max concurrent",
    )
    p.add_argument(
        "--tei-max-client-batch",
        type=int,
        default=None,
        help="Max items per TEI request (server --max-client-batch-size)",
    )
    p.add_argument(
        "--tei-max-batch-tokens",
        type=int,
        default=None,
        help="Server max batch tokens; enables token-aware packing",
    )
    p.add_argument(
        "--tei-chars-per-token",
        type=float,
        default=None,
        help="Approx chars per token for token budget estimation (default 4.0)",
    )
    p.add_argument(
        "--tei-max-input-chars",
        type=int,
        default=None,
        help="Truncate each input to this many chars (0=disabled)",
    )
    p.add_argument(
        "--tei-target-chars-per-batch",
        type=int,
        default=None,
        help="Target chars per batch if no token limit provided",
    )
    p.add_argument(
        "--tei-collapse-ws",
        type=_bool,
        default=None,
        help="Collapse whitespace before embedding (default true)",
    )
    p.add_argument(
        "--embedding-target-dim",
        type=int,
        default=None,
        help="If set, project embeddings to this dimension",
    )
    p.add_argument(
        "--embedding-projection",
        default=None,
        choices=["none", "truncate", "pad_zero"],
        help="Projection method when target dim set",
    )
    # Qdrant
    p.add_argument("--qdrant", type=_bool, default=None, help="Enable Qdrant upsert")
    p.add_argument("--qdrant-url", default=None, help="Qdrant base URL")
    p.add_argument("--qdrant-collection", default=None, help="Qdrant collection name")
    p.add_argument(
        "--qdrant-distance", default=None, help="Vector distance (Cosine|Euclid|Dot)"
    )
    p.add_argument(
        "--qdrant-vectors-name", default=None, help="Named vectors slot (optional)"
    )
    p.add_argument("--qdrant-batch", type=int, default=None, help="Upsert batch size")
    p.add_argument("--qdrant-parallel", type=int, default=None, help="Parallel upserts")
    p.add_argument(
        "--qdrant-wait", type=_bool, default=None, help="Wait until upserts are indexed"
    )
    p.add_argument("--qdrant-api-key", default=None, help="Qdrant API key (optional)")
    p.add_argument(
        "--verify-qdrant",
        action="store_true",
        help="After crawl, verify Qdrant collection count and a sample vector",
    )
    p.add_argument(
        "--qdrant-search",
        default=None,
        help="Run a Qdrant semantic search for the given query and exit",
    )
    p.add_argument(
        "--search-topk",
        type=int,
        default=5,
        help="Number of results to return for qdrant search",
    )
    p.add_argument(
        "--search-owner",
        default=None,
        help="Filter search by GitHub owner (payload.owner)",
    )
    p.add_argument(
        "--search-repo",
        default=None,
        help="Filter search by GitHub repo (payload.repo)",
    )
    p.add_argument(
        "--search-pr",
        type=int,
        default=None,
        help="Filter search by PR number (payload.pr_number)",
    )
    p.add_argument(
        "--search-item-type",
        default=None,
        help="Filter by item_type (e.g., pr_review_comment)",
    )
    p.add_argument("--search-author", default=None, help="Filter by author login")
    p.add_argument(
        "--search-path", default=None, help="Filter by file path (exact match)"
    )
    p.add_argument("--search-state", default=None, help="Filter by review_state")
    p.add_argument(
        "--search-resolved", default=None, help="Filter by resolved flag: true|false"
    )
    # Utilities
    p.add_argument(
        "--get-file-context",
        default=None,
        help="Return file snippet; use with --context-start/--context-end",
    )
    p.add_argument(
        "--context-start", type=int, default=1, help="Start line for --get-file-context"
    )
    p.add_argument(
        "--context-end", type=int, default=1, help="End line for --get-file-context"
    )
    p.add_argument(
        "--apply-suggestions",
        default=None,
        choices=["dry-run", "commit"],
        help="Apply suggestions from --items-ndjson",
    )
    p.add_argument(
        "--items-ndjson",
        default=None,
        help="Path to NDJSON items produced by --output-ndjson",
    )
    p.add_argument(
        "--only-authors",
        default=None,
        help="Comma-separated list of authors to include",
    )
    p.add_argument(
        "--only-item-types", default=None, help="Comma-separated item types to include"
    )
    p.add_argument(
        "--only-paths",
        default=None,
        help="Comma-separated file paths to include (exact)",
    )
    # Rerank
    p.add_argument(
        "--rerank", type=_bool, default=None, help="Enable reranking of search results"
    )
    p.add_argument(
        "--rerank-topk",
        type=int,
        default=None,
        help="Top results to keep after reranking",
    )
    p.add_argument(
        "--rerank-endpoint",
        default=None,
        help="Reranker base URL (defaults to TEI endpoint). Use 'local' for local GPU reranker.",
    )
    p.add_argument(
        "--rerank-model", default=None, help="Reranker model name (optional)"
    )
    p.add_argument(
        "--per-page-log",
        action="store_true",
        help="Print a line for each page: queued/completed/failed",
    )
    p.add_argument(
        "--allowed-locales",
        action="append",
        default=None,
        help="Locale codes to allow (e.g., en or en,pt). Repeatable.",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show periodic progress (pps, CPU, memory) while crawling",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows sitemap discovery and other INFO messages)",
    )
    return p


def _pick_paragraph(text: str) -> str:
    try:
        import re

        t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        paras = re.split(r"\n\s*\n+", t)
        for p in paras:
            s = p.strip()
            if not s:
                continue
            if re.search(r"(?i)navigation|search\.{3}|home page|table of contents", s):
                continue
            if s.startswith(("#", "*", "-", "[")):
                continue
            if len(s) < 80:
                continue
            link_marks = s.count("http") + s.count("](")
            if link_marks >= 3 and (
                len(re.sub(r"https?://\S+|\[[^\]]*\]\([^)]*\)", "", s)) < 80
            ):
                continue
            return re.sub(r"\s+", " ", s).strip()
        # fallback
        lines = [
            ln.strip()
            for ln in (t or "").splitlines()
            if len(ln.strip()) >= 60 and not ln.strip().startswith(("#", "*", "-", "["))
        ]
        import re as _re

        return _re.sub(r"\s+", " ", lines[0]).strip() if lines else ""
    except Exception:
        return ""


async def _run(args: argparse.Namespace) -> int:
    if not getattr(args, "qdrant_search", None) and not args.url:
        print(
            "optimized-crawler: error: --url is required unless using --qdrant-search",
            file=sys.stderr,
        )
        return 2

    # Configure logging to show sitemap discovery success messages
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(message)s",  # Simple format to avoid duplicating our emoji formatting
        stream=sys.stderr,
    )

    # Configure the url_discovery logger to always show success messages at WARNING level
    # This ensures sitemap discovery messages are visible even without --verbose
    url_discovery_logger = logging.getLogger("url_discovery")
    url_discovery_logger.setLevel(logging.INFO)

    if args.verbose:
        print(
            "🔍 Verbose logging enabled - showing all discovery messages",
            file=sys.stderr,
        )

    # Load .env (without external deps) to allow environment-based config
    def _load_env() -> None:
        import pathlib

        candidates = [
            pathlib.Path(".env"),
            pathlib.Path(".env.local"),
            pathlib.Path("../.env"),
            pathlib.Path("../../.env"),
        ]
        for p in candidates:
            try:
                if p.exists() and p.is_file():
                    for line in p.read_text(encoding="utf-8").splitlines():
                        s = line.strip()
                        if not s or s.startswith("#"):
                            continue
                        if "=" not in s:
                            continue
                        k, v = s.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                continue

    _load_env()

    # Create overrides dict from CLI arguments
    from ..settings import get_settings

    settings = get_settings()

    overrides = {
        "max_urls_to_discover": args.max_urls,
        "max_concurrent_crawls": args.concurrency,
        "content_validation": args.content_validation,
        "page_timeout": max(1, int(args.page_timeout_ms / 1000)),
        "output_dir": args.output_dir,
    }

    if args.doc_relax:
        overrides["doc_relax_validation_patterns"] = list(args.doc_relax)

    # Apply performance toggles
    if args.aggressive:
        overrides.update(
            {
                "max_concurrent_crawls": max(args.concurrency, 20),
                "memory_threshold_percent": 85,
                "page_timeout": 15,  # seconds
            }
        )

    if args.preset_13700k:
        # Tailored for i7-13700K class machines
        overrides.update(
            {
                "max_concurrent_crawls": max(args.concurrency, 28),
                "memory_threshold_percent": 85,
                "page_timeout": min(
                    int(args.page_timeout_ms) // 1000, 15
                ),  # convert ms to seconds
            }
        )

    if args.no_cache:
        overrides["cache_enabled"] = False

    if args.javascript is not None:
        overrides["browser_mode"] = "full" if bool(args.javascript) else "text"

    # JS retry removed - using browser_mode configuration instead

    # Embeddings configuration
    if args.embeddings is not None:
        overrides["enable_embeddings"] = bool(args.embeddings)
    if args.tei_endpoint:
        overrides["tei_endpoint"] = args.tei_endpoint
    if args.tei_batch is not None:
        overrides["tei_batch_size"] = int(args.tei_batch)
    if args.tei_parallel is not None:
        overrides["tei_parallel_requests"] = max(1, int(args.tei_parallel))
    if args.tei_timeout_s is not None:
        overrides["tei_timeout_s"] = float(args.tei_timeout_s)
    if args.tei_max_concurrent is not None:
        overrides["tei_max_concurrent_requests"] = max(1, int(args.tei_max_concurrent))
    if args.tei_max_client_batch is not None:
        overrides["tei_max_client_batch_size"] = max(1, int(args.tei_max_client_batch))
    if args.tei_max_batch_tokens is not None:
        overrides["tei_max_batch_tokens"] = max(0, int(args.tei_max_batch_tokens))
    if args.tei_chars_per_token is not None:
        overrides["tei_chars_per_token"] = max(0.1, float(args.tei_chars_per_token))
    if args.tei_max_input_chars is not None:
        overrides["tei_max_input_chars"] = max(0, int(args.tei_max_input_chars))
    if args.tei_target_chars_per_batch is not None:
        overrides["tei_target_chars_per_batch"] = max(
            1000, int(args.tei_target_chars_per_batch)
        )
    if args.tei_collapse_ws is not None:
        overrides["tei_collapse_whitespace"] = bool(args.tei_collapse_ws)
    if args.embedding_target_dim is not None:
        overrides["embedding_target_dim"] = max(0, int(args.embedding_target_dim))
    if args.embedding_projection is not None:
        overrides["embedding_projection"] = args.embedding_projection

    # Qdrant configuration
    if args.qdrant is not None:
        overrides["enable_qdrant"] = bool(args.qdrant)
    if args.qdrant_url:
        overrides["qdrant_url"] = args.qdrant_url
    if args.qdrant_collection:
        overrides["qdrant_collection"] = args.qdrant_collection
    if args.qdrant_distance:
        overrides["qdrant_distance"] = args.qdrant_distance
    if args.qdrant_vectors_name is not None:
        overrides["qdrant_vectors_name"] = args.qdrant_vectors_name or ""
    if args.qdrant_batch is not None:
        overrides["qdrant_batch_size"] = int(args.qdrant_batch)
    if args.qdrant_parallel is not None:
        overrides["qdrant_parallel_requests"] = max(1, int(args.qdrant_parallel))
    if args.qdrant_wait is not None:
        overrides["qdrant_upsert_wait"] = bool(args.qdrant_wait)
    if args.qdrant_api_key:
        overrides["qdrant_api_key"] = args.qdrant_api_key

    # Language filtering
    if args.allowed_locales is not None:
        locales: list[str] = []
        for entry in args.allowed_locales:
            if entry is None:
                continue
            e = entry.strip()
            if not e:
                continue
            if e.lower() in ("*", "all"):
                locales = []
                break
            locales.extend([p.strip() for p in e.split(",") if p.strip()])
        overrides["allowed_locales"] = [loc.lower() for loc in locales]

    # Subcommand: Qdrant semantic search
    if getattr(args, "qdrant_search", None):
        query = str(args.qdrant_search)
        # Allow env to override topk default
        try:
            topk_env = int(os.environ.get("OPTIMIZED_CRAWLER_SEARCH_TOPK", "0"))
        except Exception:
            topk_env = 0
        topk = int(getattr(args, "search_topk", 5) or (topk_env or 5))
        from ..clients.local_reranker import LocalReranker
        from ..clients.qdrant_http_client import QdrantClient
        from ..clients.tei_client import TEIEmbeddingsClient

        async with TEIEmbeddingsClient(
            overrides.get("tei_endpoint", settings.tei_url),
            model=(overrides.get("tei_model_name", settings.tei_model) or None),
            timeout_s=max(
                5.0, float(overrides.get("tei_timeout_s", settings.tei_timeout))
            ),
        ) as tei:
            vecs = await tei.embed_texts([query])
        if not vecs or not isinstance(vecs[0], list):
            print("Failed to embed query", flush=True)
            return 1
        qvec = vecs[0]
        # Optional structured filter on owner/repo/pr_number + extra fields
        qfilter = None
        must = []
        if args.search_owner:
            must.append({"key": "owner", "match": {"value": str(args.search_owner)}})
        if args.search_repo:
            must.append({"key": "repo", "match": {"value": str(args.search_repo)}})
        if args.search_pr is not None:
            must.append({"key": "pr_number", "match": {"value": int(args.search_pr)}})
        if args.search_item_type:
            must.append(
                {"key": "item_type", "match": {"value": str(args.search_item_type)}}
            )
        if args.search_author:
            must.append({"key": "author", "match": {"value": str(args.search_author)}})
        if args.search_path:
            must.append({"key": "path", "match": {"value": str(args.search_path)}})
        if args.search_state:
            must.append(
                {"key": "review_state", "match": {"value": str(args.search_state)}}
            )
        if args.search_resolved is not None:
            val = str(args.search_resolved).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            must.append({"key": "resolved", "match": {"value": bool(val)}})
        if must:
            qfilter = {"must": must}

        async with QdrantClient(
            overrides.get("qdrant_url", settings.qdrant_url),
            api_key=overrides.get("qdrant_api_key", settings.qdrant_api_key),
            timeout_s=15.0,
        ) as qc:
            res = await qc.search(
                overrides.get("qdrant_collection", settings.qdrant_collection),
                vector=qvec,
                limit=topk,
                with_payload=True,
                with_vectors=False,
                query_filter=qfilter,
            )
        hits = res.get("result", []) if isinstance(res, dict) else []

        # Optional rerank step (TEI or local GPU)
        # Default behavior: if not explicitly set, auto-enable rerank when CUDA is available (local reranker).
        auto_cuda = False
        try:
            from ..clients.local_reranker import _has_cuda as _rr_has_cuda

            auto_cuda = bool(_rr_has_cuda())
        except Exception:
            auto_cuda = False

        if args.rerank is not None:
            do_rerank = bool(args.rerank)
        elif overrides.get("enable_rerank", settings.reranker_enabled):
            do_rerank = True
        else:
            do_rerank = auto_cuda  # auto default to local GPU when available

        if do_rerank and hits:
            rerank_topk = int(args.rerank_topk or overrides.get("rerank_topk", 5) or 5)
            # endpoint resolution: arg > cfg > auto('local' if cuda else '')
            rerank_ep = (
                args.rerank_endpoint
                or overrides.get("rerank_endpoint", "")
                or ("local" if auto_cuda else "")
            )
            rerank_model = (
                args.rerank_model
                or overrides.get("rerank_model_name", settings.reranker_model)
                or ""
            )
            docs = []
            for h in hits:
                payload = h.get("payload", {}) or {}
                title = payload.get("title", "")
                para = _pick_paragraph(str(payload.get("text", "")))
                docs.append(
                    (title + "\n\n" + para).strip()
                    or title
                    or str(payload.get("url", ""))
                )

            try:
                # Only local reranker is supported; fall back to local if a non-local endpoint is provided.
                if rerank_ep and rerank_ep.lower() != "local":
                    print(
                        f"Rerank endpoint '{rerank_ep}' not supported; falling back to local reranker."
                    )
                rr = LocalReranker(
                    model_name=(rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
                )
                rr_scores = rr.rerank(query, docs, top_n=rerank_topk)
                print(f"Reranker: local '{rr.model_name}' on device {rr.get_device()}")
                score_map = {
                    it["index"]: float(it.get("score", 0.0)) for it in rr_scores
                }
                indexed = list(enumerate(hits))
                indexed.sort(key=lambda it: score_map.get(it[0], 0.0), reverse=True)
                hits = [h for _, h in indexed][:rerank_topk]
            except Exception as e:
                print(f"Rerank failed: {e}")

        print("Search Results" + (" (reranked)" if do_rerank else ""))
        for i, h in enumerate(hits, 1):
            payload = h.get("payload", {}) or {}
            url = payload.get("url", "")
            title = payload.get("title", "")
            para = _pick_paragraph(str(payload.get("text", "")))
            score = h.get("score", 0)
            print(f"{i}. [{score:.4f}] {title}\n   {url}\n   {para[:400]}\n")
        return 0

    # Subcommand: get file context
    if getattr(args, "get_file_context", None):
        path = str(args.get_file_context)
        start = int(getattr(args, "context_start", 1) or 1)
        end = int(getattr(args, "context_end", start) or start)
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
            s = max(1, start)
            e = max(s, end)
            snippet = "\n".join(lines[s - 1 : e])
            print(snippet)
            return 0
        except Exception as e:
            print(f"Failed to read file context: {e}")
            return 1

    # Subcommand: apply suggestions from NDJSON items
    if getattr(args, "apply_suggestions", None):
        mode = str(args.apply_suggestions).lower()
        items_path = str(getattr(args, "items_ndjson", "") or "")
        if not items_path:
            print("--items-ndjson is required for --apply-suggestions", flush=True)
            return 2
        # Filters
        only_authors = {
            s.strip()
            for s in (getattr(args, "only_authors", "") or "").split(",")
            if s.strip()
        }
        only_types = {
            s.strip()
            for s in (getattr(args, "only_item_types", "") or "").split(",")
            if s.strip()
        }
        only_paths = {
            s.strip()
            for s in (getattr(args, "only_paths", "") or "").split(",")
            if s.strip()
        }
        import json as _json

        from ..utils.github_suggestions import (
            apply_suggestion as _apply_sugg,
        )
        from ..utils.github_suggestions import (
            extract_suggestions,
        )
        from ..utils.github_suggestions import (
            unified_diff as _udiff,
        )

        changes: list[tuple[str, str, str]] = []  # (path, before, after)
        try:
            with open(items_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        item = _json.loads(line)
                    except Exception:
                        continue
                    md = item.get("metadata") or {}
                    itype = str(md.get("item_type", ""))
                    if only_types and itype not in only_types:
                        continue
                    author = str(md.get("author", ""))
                    if only_authors and author not in only_authors:
                        continue
                    path = str(md.get("path", ""))
                    if only_paths and path and (path not in only_paths):
                        continue
                    content = str(item.get("content", ""))
                    suggs = extract_suggestions(content)
                    if not suggs:
                        continue
                    if not path:
                        continue  # cannot apply without path
                    start_line = int(md.get("start_line") or md.get("line") or 1)
                    end_line = int(md.get("end_line") or md.get("line") or start_line)
                    try:
                        with open(path, encoding="utf-8") as _f:
                            before = _f.read()
                    except Exception:
                        continue
                    after = before
                    for s in suggs:
                        after = _apply_sugg(
                            after,
                            start_line=start_line,
                            end_line=end_line,
                            suggestion_text=s.content,
                        )
                    if after != before:
                        changes.append((path, before, after))
        except Exception as e:
            print(f"Failed to read items: {e}")
            return 1
        if not changes:
            print("No applicable suggestions found.")
            return 0
        # Report diff or write files
        total_changed = 0
        for path, before, after in changes:
            diff = _udiff(path, before, after)
            if mode == "dry-run":
                print(diff)
            elif mode == "commit":
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(after)
                    total_changed += 1
                except Exception as e:
                    print(f"Failed to write {path}: {e}")
        if mode == "commit" and total_changed:
            # Stage and commit
            try:
                import subprocess as sp

                sp.run(["git", "add", "-A"], check=False)
                sp.run(
                    ["git", "commit", "-m", "chore: apply GitHub review suggestions"],
                    check=False,
                )
            except Exception:
                pass
        return 0

    # Resolve progress preference from env if flag not passed
    def _env_bool(name: str, default: bool = False) -> bool:
        try:
            v = os.environ.get(name)
            if v is None:
                return default
            return v.strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            return default

    show_progress = bool(
        args.progress or _env_bool("OPTIMIZED_CRAWLER_PROGRESS", False)
    )

    # Initialize output and log managers
    output_mgr = OutputManager(args.output_dir)
    log_mgr = LogManager("./logs")

    # Setup logging
    crawl_logger = log_mgr.setup_crawl_logger()
    error_logger = log_mgr.setup_error_logger()
    console_logger = log_mgr.setup_console_logger()

    # Keep url_discovery visible after handler reset
    url_discovery_logger = logging.getLogger("url_discovery")
    url_discovery_logger.propagate = False
    for h in console_logger.handlers:
        if h not in url_discovery_logger.handlers:
            url_discovery_logger.addHandler(h)
    if getattr(args, "stream", False):
        console_logger.warning(
            "Flag --stream is deprecated; setting ENABLE_STREAMING=1 for this run."
        )
        os.environ.setdefault("ENABLE_STREAMING", "1")

    # Clean outputs if requested
    if args.clean_outputs:
        console_logger.info("Cleaning %s directory...", args.output_dir)
        output_mgr.cleanup_old_outputs()
        output_mgr.clean_cache()

    # Get domain for output management
    domain = output_mgr.sanitize_domain(args.url)

    # Rotate backup before crawl (unless --no-backup)
    if not args.no_backup:
        output_mgr.rotate_crawl_backup(domain)

    crawl_logger.info(f"Starting crawl for {args.url}")
    crawl_logger.info(f"Output directory: {args.output_dir}")
    crawl_logger.info(f"Domain: {domain}")

    strat = CrawlOrchestrator(settings, overrides)

    # Optional: human-friendly per-page logs via monitoring hooks
    if args.per_page_log or show_progress:

        def _print(s: str) -> None:
            import contextlib

            with contextlib.suppress(Exception):
                print(s, flush=True)

        def on_crawl_started(urls=None, metrics=None, **_):  # type: ignore[no-redef]
            if not urls:
                return
            for u in urls:
                _print(f"⇢ fetching {u}")

        def on_page_crawled(url=None, content_length=0, crawl_time=0.0, **_):  # type: ignore[no-redef]
            ms = int(float(crawl_time) * 1000) if crawl_time else 0
            _print(f"✅ complete {url} [{content_length} bytes, {ms} ms]")

        def on_page_failed(url=None, error=None, error_type=None, **_):  # type: ignore[no-redef]
            et = f" ({error_type})" if error_type else ""
            _print(f"❌ failed   {url}{et}: {error}")

        def on_hash_placeholder_detected(url=None, **_):  # type: ignore[no-redef]
            _print(f"⚠ placeholder {url}")

        strat.set_hook("crawl_started", on_crawl_started)
        strat.set_hook("page_crawled", on_page_crawled)
        strat.set_hook("page_failed", on_page_failed)
        strat.set_hook("hash_placeholder_detected", on_hash_placeholder_detected)

        if show_progress:
            import time as _t

            _last = {"t": 0.0}

            def on_performance_sample(
                memory_mb=None,
                memory_percent=None,
                cpu_percent=None,
                pages_per_second=None,
                metrics=None,
                **_,
            ):  # type: ignore[no-redef]
                try:
                    now = _t.time()
                    if now - _last["t"] < 2.0:
                        return
                    _last["t"] = now
                    pc = int(metrics.pages_crawled) if metrics else 0
                    pf = int(metrics.pages_failed) if metrics else 0
                    ud = int(metrics.urls_discovered) if metrics else 0
                    processed = pc + pf
                    # Compute processed rate over wall time for a more responsive view
                    if metrics and metrics.start_time:
                        elapsed = max(0.1, now - float(metrics.start_time))
                        pps = processed / elapsed
                    else:
                        pps = float(pages_per_second or 0.0)
                    cpu = float(cpu_percent or 0.0)
                    mem = float(memory_percent or 0.0)
                    _print(
                        f"… progress: {processed}/{ud} (ok={pc}, fail={pf}), {pps:.2f} p/s, CPU {cpu:.1f}%, MEM {mem:.1f}%"
                    )
                except Exception:
                    pass

            strat.set_hook("performance_sample", on_performance_sample)
    await strat.start()
    try:
        # Streaming is controlled by ENABLE_STREAMING environment variable (or --stream)
        crawl_logger.info(f"Beginning crawl of {args.url}")
        resp = await strat.crawl(args.url)
        crawl_logger.info("Crawl completed successfully")

        # Save outputs using OutputManager (unless --skip-output)
        if not args.skip_output:
            try:
                pages = strat.get_last_pages() or []

                # Prepare page data with embeddings if requested
                page_data = []
                for pg in pages:
                    obj: dict[str, Any] = {
                        "url": getattr(pg, "url", ""),
                        "title": getattr(pg, "title", ""),
                        "word_count": getattr(pg, "word_count", 0),
                        "links": getattr(pg, "links", []),
                        "images": getattr(pg, "images", []),
                        "metadata": getattr(pg, "metadata", {}),
                        "content": getattr(pg, "content", ""),
                    }
                    page_data.append(obj)

                # Get performance report
                report = strat.get_performance_report() or {}

                # Save all outputs
                output_mgr.save_crawl_outputs(domain, resp.html, page_data, report)

                # Update index with metadata
                output_mgr.update_index(
                    domain,
                    {
                        "url": args.url,
                        "pages_crawled": len(page_data),
                        "success": True,
                        "concurrency": args.concurrency,
                        "max_urls": args.max_urls,
                    },
                )

                crawl_logger.info(
                    f"Saved outputs to {args.output_dir}/crawls/{domain}/latest/"
                )
                console_logger.info(
                    f"✅ Outputs saved to: {args.output_dir}/crawls/{domain}/latest/"
                )

            except Exception as e:
                error_logger.error(f"Failed to save outputs: {e}")
                console_logger.error(f"⚠️ Failed to save outputs: {e}")
        else:
            crawl_logger.info("Skipping output save (--skip-output)")
            console_logger.info("📝 Outputs not saved (--skip-output)")

        # Optional: verify Qdrant contents
        if overrides.get("enable_qdrant", False) and args.verify_qdrant:
            try:
                from ..clients.qdrant_http_client import QdrantClient

                qurl = overrides.get("qdrant_url", settings.qdrant_url)
                qcol = overrides.get("qdrant_collection", settings.qdrant_collection)
                qkey = overrides.get("qdrant_api_key", settings.qdrant_api_key)
                async with QdrantClient(qurl, api_key=qkey, timeout_s=15.0) as qc:
                    info = await qc.get_collection(qcol)
                    cnt = await qc.count_points(qcol, exact=False)
                    sample = await qc.scroll_points(
                        qcol, limit=1, with_vectors=True, with_payload=True
                    )
                print("\nQdrant Verification")
                print("- Collection:", qcol)
                try:
                    vectors_info = (
                        info.get("result", {})
                        .get("config", {})
                        .get("params", {})
                        .get("vectors", {})
                    )
                except Exception:
                    vectors_info = {}
                if isinstance(vectors_info, dict) and "size" in vectors_info:
                    print("- Vector Size:", vectors_info.get("size"))
                    print("- Distance:", vectors_info.get("distance"))
                else:
                    # named vectors variant
                    try:
                        name, cfgv = next(iter(vectors_info.items()))
                        print("- Vectors Name:", name)
                        print("- Vector Size:", cfgv.get("size"))
                        print("- Distance:", cfgv.get("distance"))
                    except Exception:
                        pass
                print("- Points (approx):", cnt)
                try:
                    pts = sample.get("result", {}).get("points", [])
                    if pts:
                        p0 = pts[0]
                        vec = p0.get("vector")
                        vlen = (
                            len(vec)
                            if isinstance(vec, list)
                            else (
                                len(next(iter(vec.values())))
                                if isinstance(vec, dict)
                                else 0
                            )
                        )
                        print("- Sample ID:", p0.get("id"))
                        print("- Sample Vector Length:", vlen)
                        payload = p0.get("payload", {}) or {}
                        print("- Sample URL:", payload.get("url", ""))
                    else:
                        print("- Sample: no points returned")
                except Exception:
                    pass
            except Exception as e:
                print("Qdrant verification failed:", e)

        # Enhanced terminal summary (factored out)
        headers = resp.response_headers or {}
        report = strat.get_performance_report() or {}
        from ..shared.reporting import print_enhanced_report

        report = print_enhanced_report(args, headers, report, strat)
        # Skip legacy inline reporter below
        # (region intentionally left blank after refactor)
        #

        def _fmt_pct(x: float | int | str | None) -> str:
            try:
                return f"{float(x):.1f}%"
            except Exception:
                return "n/a"

        def _fmt_num(x: str | None) -> str:
            try:
                # some headers are strings
                return f"{int(float(x)):,}"
            except Exception:
                return str(x or "0")

        # Color helpers
        def _supports_color() -> bool:
            try:
                return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
            except Exception:
                return False

        _COLOR = _supports_color()

        def _sty(s: str, code: str) -> str:
            return f"\033[{code}m{s}\033[0m" if _COLOR else s

        def H(s: str) -> str:  # Header  # noqa: N802
            return _sty(s, "1;36")

        def KEY(s: str) -> str:  # Label  # noqa: N802
            return _sty(s, "90")

        def VAL(s: str) -> str:  # Value  # noqa: N802
            return _sty(s, "36")

        # Legacy inline helpers and header parsing removed; reporting handled by shared module
        # JSON reports are now saved automatically by OutputManager
        return 0
    except Exception as e:
        error_logger.error(f"Crawl failed: {e}", exc_info=True)
        console_logger.error(f"❌ Crawl failed: {e}")

        # Try to save partial results if available
        if not args.skip_output:
            try:
                pages = strat.get_last_pages() or []
                if pages:
                    report = strat.get_performance_report() or {}
                    output_mgr.save_crawl_outputs(domain, None, pages, report)
                    output_mgr.update_index(
                        domain,
                        {
                            "url": args.url,
                            "pages_crawled": len(pages),
                            "success": False,
                            "error": str(e),
                        },
                    )
                    console_logger.info("💾 Saved partial results")
            except Exception as save_error:
                error_logger.error(f"Failed to save partial results: {save_error}")

        return 1
    finally:
        await strat.close()


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
