"""
CLI runner for the optimized crawler.

Usage:
  uv run python -m crawler_mcp.crawlers.optimized.run \
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
import json
import logging
import os
import sys
from typing import Any

from . import OptimizedConfig, OptimizedCrawlerStrategy


def _bool(x: str) -> bool:
    return x.lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
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
        "--output-html",
        default="crawl_all.html",
        help="Path to write the combined HTML",
    )
    p.add_argument(
        "--output-ndjson",
        default="",
        help="Path to write per-page NDJSON (one JSON object per line)",
    )
    p.add_argument(
        "--report-json",
        default="",
        help="Path to write the performance report JSON",
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
        help="Stream results (lower latency, same final output)",
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
    p.add_argument(
        "--ndjson-include-embeddings",
        action="store_true",
        help="Include embeddings array in NDJSON output (large)",
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

    # Start from env-driven config, then apply CLI overrides
    cfg = OptimizedConfig.from_env()
    cfg.max_urls_to_discover = args.max_urls
    cfg.max_concurrent_crawls = args.concurrency
    cfg.content_validation = args.content_validation
    cfg.page_timeout = int(args.page_timeout_ms)
    if args.doc_relax:
        cfg.doc_relax_validation_patterns = list(args.doc_relax)

    # Apply performance toggles
    if args.aggressive:
        cfg = cfg.get_aggressive_config()

    if args.preset_13700k:
        # Tailored for i7-13700K class machines
        cfg = cfg.get_aggressive_config()
        cfg.max_concurrent_crawls = max(cfg.max_concurrent_crawls, 28)
        cfg.memory_threshold = max(cfg.memory_threshold, 85.0)
        cfg.page_timeout = min(cfg.page_timeout, 15000)

    if args.no_cache:
        cfg.enable_cache = False

    if args.javascript is not None:
        cfg.javascript_enabled = bool(args.javascript)

    if args.no_js_retry:
        cfg.js_retry_enabled = False

    # Embeddings configuration
    if args.embeddings is not None:
        cfg.enable_embeddings = bool(args.embeddings)
    if args.tei_endpoint:
        cfg.tei_endpoint = args.tei_endpoint
    if args.tei_batch is not None:
        cfg.tei_batch_size = int(args.tei_batch)
    if args.tei_parallel is not None:
        cfg.tei_parallel_requests = max(1, int(args.tei_parallel))
    if args.tei_timeout_s is not None:
        cfg.tei_timeout_s = float(args.tei_timeout_s)
    if args.tei_max_concurrent is not None:
        cfg.tei_max_concurrent_requests = max(1, int(args.tei_max_concurrent))
    if args.tei_max_client_batch is not None:
        cfg.tei_max_client_batch_size = max(1, int(args.tei_max_client_batch))
    if args.tei_max_batch_tokens is not None:
        cfg.tei_max_batch_tokens = max(0, int(args.tei_max_batch_tokens))
    if args.tei_chars_per_token is not None:
        cfg.tei_chars_per_token = max(0.1, float(args.tei_chars_per_token))
    if args.tei_max_input_chars is not None:
        cfg.tei_max_input_chars = max(0, int(args.tei_max_input_chars))
    if args.tei_target_chars_per_batch is not None:
        cfg.tei_target_chars_per_batch = max(1000, int(args.tei_target_chars_per_batch))
    if args.tei_collapse_ws is not None:
        cfg.tei_collapse_whitespace = bool(args.tei_collapse_ws)
    if args.embedding_target_dim is not None:
        cfg.embedding_target_dim = max(0, int(args.embedding_target_dim))
    if args.embedding_projection is not None:
        cfg.embedding_projection = args.embedding_projection

    # Qdrant configuration
    if args.qdrant is not None:
        cfg.enable_qdrant = bool(args.qdrant)
    if args.qdrant_url:
        cfg.qdrant_url = args.qdrant_url
    if args.qdrant_collection:
        cfg.qdrant_collection = args.qdrant_collection
    if args.qdrant_distance:
        cfg.qdrant_distance = args.qdrant_distance
    if args.qdrant_vectors_name is not None:
        cfg.qdrant_vectors_name = args.qdrant_vectors_name or ""
    if args.qdrant_batch is not None:
        cfg.qdrant_batch_size = int(args.qdrant_batch)
    if args.qdrant_parallel is not None:
        cfg.qdrant_parallel_requests = max(1, int(args.qdrant_parallel))
    if args.qdrant_wait is not None:
        cfg.qdrant_upsert_wait = bool(args.qdrant_wait)
    if args.qdrant_api_key:
        cfg.qdrant_api_key = args.qdrant_api_key

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
        cfg.allowed_locales = [l.lower() for l in locales]

    # Subcommand: Qdrant semantic search
    if getattr(args, "qdrant_search", None):
        query = str(args.qdrant_search)
        # Allow env to override topk default
        try:
            topk_env = int(os.environ.get("OPTIMIZED_CRAWLER_SEARCH_TOPK", "0"))
        except Exception:
            topk_env = 0
        topk = int(getattr(args, "search_topk", 5) or (topk_env or 5))
        from .local_reranker import LocalReranker
        from .qdrant_http_client import QdrantClient
        from .tei_client import TEIEmbeddingsClient

        async with TEIEmbeddingsClient(
            cfg.tei_endpoint,
            model=(cfg.tei_model_name or None),
            timeout_s=max(5.0, float(cfg.tei_timeout_s)),
        ) as tei:
            vecs = await tei.embed_texts([query])
        if not vecs or not isinstance(vecs[0], list):
            print("Failed to embed query", flush=True)
            return 1
        qvec = vecs[0]
        # Optional structured filter on owner/repo/pr_number
        qfilter = None
        if args.search_owner or args.search_repo or (args.search_pr is not None):
            must = []
            if args.search_owner:
                must.append(
                    {"key": "owner", "match": {"value": str(args.search_owner)}}
                )
            if args.search_repo:
                must.append({"key": "repo", "match": {"value": str(args.search_repo)}})
            if args.search_pr is not None:
                must.append(
                    {"key": "pr_number", "match": {"value": int(args.search_pr)}}
                )
            if must:
                qfilter = {"must": must}

        async with QdrantClient(
            cfg.qdrant_url, api_key=cfg.qdrant_api_key, timeout_s=15.0
        ) as qc:
            res = await qc.search(
                cfg.qdrant_collection,
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
            from .local_reranker import _has_cuda as _rr_has_cuda

            auto_cuda = bool(_rr_has_cuda())
        except Exception:
            auto_cuda = False

        if args.rerank is not None:
            do_rerank = bool(args.rerank)
        elif getattr(cfg, "enable_rerank", False):
            do_rerank = True
        else:
            do_rerank = auto_cuda  # auto default to local GPU when available

        if do_rerank and hits:
            rerank_topk = int(args.rerank_topk or getattr(cfg, "rerank_topk", 5) or 5)
            # endpoint resolution: arg > cfg > auto('local' if cuda else '')
            rerank_ep = (
                args.rerank_endpoint
                or getattr(cfg, "rerank_endpoint", "")
                or ("local" if auto_cuda else "")
            )
            rerank_model = (
                args.rerank_model or getattr(cfg, "rerank_model_name", "") or ""
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

    strat = OptimizedCrawlerStrategy(cfg)

    # Optional: human-friendly per-page logs via monitoring hooks
    if args.per_page_log or show_progress:

        def _print(s: str) -> None:
            try:
                print(s, flush=True)
            except Exception:
                pass

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
        # Force streaming when per-page logging is requested to reduce latency
        resp = await strat.crawl(
            args.url, stream=(args.stream or args.per_page_log or show_progress)
        )

        # Write combined HTML
        if args.output_html:
            with open(args.output_html, "w", encoding="utf-8") as f:
                f.write(resp.html or "")

        # Dump per-page NDJSON if requested
        if args.output_ndjson:
            pages = strat.get_last_pages()
            with open(args.output_ndjson, "w", encoding="utf-8") as f:
                for pg in pages or []:
                    obj: dict[str, Any] = {
                        "url": getattr(pg, "url", ""),
                        "title": getattr(pg, "title", ""),
                        "word_count": getattr(pg, "word_count", 0),
                        "links": getattr(pg, "links", []),
                        "images": getattr(pg, "images", []),
                        "metadata": getattr(pg, "metadata", {}),
                        "content": getattr(pg, "content", ""),
                    }
                    if args.ndjson_include_embeddings:
                        emb = (obj.get("metadata", {}) or {}).get("embedding")
                        if emb is None:
                            # keep explicit null to signal disabled/omitted
                            obj["embedding"] = None
                        else:
                            obj["embedding"] = emb
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Optional: verify Qdrant contents
        if cfg.enable_qdrant and args.verify_qdrant:
            try:
                from .qdrant_http_client import QdrantClient

                qurl = cfg.qdrant_url
                qcol = cfg.qdrant_collection
                qkey = cfg.qdrant_api_key
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

        # Enhanced terminal summary (with color by default when TTY)
        headers = resp.response_headers or {}
        report = strat.get_performance_report() or {}

        def _hdr(name: str, default: str = "") -> str:
            return str(headers.get(name, default) or default)

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

        def H(s: str) -> str:  # Header
            return _sty(s, "1;36")

        def KEY(s: str) -> str:  # Label
            return _sty(s, "90")

        def VAL(s: str) -> str:  # Value
            return _sty(s, "36")

        def GOOD(s: str) -> str:
            return _sty(s, "32")

        def BAD(s: str) -> str:
            return _sty(s, "31")

        def WARN(s: str) -> str:
            return _sty(s, "33")

        success = _hdr("X-Crawl-Success", "False").lower() == "true"
        pages_crawled = _hdr("X-Pages-Crawled", "0")
        urls_discovered = _hdr("X-URLs-Discovered", "0")
        pps = _hdr("X-Pages-Per-Second", "0")
        total_h = _hdr("X-Total-Content-Human", "-")
        avg_h = _hdr("X-Average-Page-Size-Human", "-")
        failed_count = _hdr("X-Failed-URLs-Count", "0")
        failed_sample = _hdr("X-Failed-URLs-Sample", "")
        placeholders = _hdr("X-Placeholders-Filtered", "0")
        total_links = _hdr("X-Total-Links", "0")
        sample_links = _hdr("X-Sample-Links", "")

        # Section: Header
        line = "".ljust(70, "=")
        print(_sty(line, "90"))
        print(H(" 🚀 Optimized Crawl Report "))
        print(_sty(line, "90"))

        # Section: Crawl Summary
        print("\n" + H("📊 Crawl Summary"))
        print(KEY("- 🔗 URL:"), VAL(args.url))
        print(KEY("- ✅ Success:"), GOOD("✅ True") if success else BAD("❌ False"))
        print(KEY("- 📄 Pages:"), VAL(f"{pages_crawled} / {urls_discovered}"))
        print(KEY("- ⚡ Throughput:"), VAL(f"{pps} pages/sec"))
        print(KEY("- 📦 Total Content:"), VAL(total_h))
        print(KEY("- 📏 Avg Page Size:"), VAL(avg_h))
        ph = (
            WARN(placeholders)
            if placeholders and placeholders != "0"
            else KEY(placeholders)
        )
        print(KEY("- 🧩 Placeholders Filtered:"), ph)
        fc = (
            BAD(failed_count)
            if failed_count and failed_count != "0"
            else KEY(failed_count)
        )
        print(KEY("- ❗ Failed URLs:"), fc)

        # Section: Content & Links
        print("\n" + H("🔗 Content & Links"))
        # Derive richer link stats from pages when available
        pages = strat.get_last_pages() or []
        total_link_instances = 0
        unique_link_set: set[str] = set()
        host_counts: dict[str, int] = {}
        try:
            from urllib.parse import urlparse as _urlparse
        except Exception:  # pragma: no cover
            _urlparse = None

        for pg in pages:
            try:
                page_links = getattr(pg, "links", []) or []
                total_link_instances += len(page_links)
                for l in page_links:
                    if not l:
                        continue
                    unique_link_set.add(l)
                    if _urlparse:
                        try:
                            host = _urlparse(l).netloc
                            if host:
                                host_counts[host] = host_counts.get(host, 0) + 1
                        except Exception:
                            pass
            except Exception:
                continue

        unique_links_count = len(unique_link_set)
        print(
            KEY("- 🔗 Unique Links:"),
            VAL(_fmt_num(str(unique_links_count or total_links))),
        )
        print(KEY("- 🔁 Link Instances:"), VAL(_fmt_num(str(total_link_instances))))
        if sample_links:
            links_list = [s for s in sample_links.split(",") if s][:5]
            if links_list:
                print(KEY("- 🔎 Sample Links:"))
                for i, link in enumerate(links_list, 1):
                    print(VAL(f"  🔹 {i}. {link}"))
        # Top link hosts
        if host_counts:
            print(KEY("- 🌐 Top Link Hosts:"))
            for i, (host, cnt) in enumerate(
                sorted(host_counts.items(), key=lambda x: (-x[1], x[0]))[:5], 1
            ):
                print(VAL(f"  {i}. {host} ({cnt})"))

        # Section: Failures (if any)
        if failed_count and failed_count != "0":
            print("\n" + H("💥 Failures"))
            if failed_sample:
                for i, link in enumerate(
                    [s for s in failed_sample.split(",") if s][:5], 1
                ):
                    print(BAD(f"  🔻 {i}. {link}"))

        # Section: Validation & Quality
        val = report.get("validation_summary", {}) or {}
        relaxed_total = val.get("relaxed_acceptances_total") or val.get(
            "relaxed_acceptances"
        )
        inv_reasons = val.get("invalid_reasons", {}) or val.get(
            "invalid_reason_counts", {}
        )
        relaxed_reasons = val.get("relaxed_acceptance_reasons", {}) or val.get(
            "relaxed_acceptance_reason_counts", {}
        )
        content_validations = val.get("content_validations_recorded")
        if any([relaxed_total, inv_reasons, relaxed_reasons, content_validations]):
            print("\n" + H("🧪 Validation"))
            if content_validations is not None:
                print(KEY("- 🧮 Validations Recorded:"), VAL(str(content_validations)))
            if relaxed_total is not None:
                print(KEY("- 🪶 Relaxed Acceptances:"), VAL(str(relaxed_total)))
            if inv_reasons:
                print(KEY("- 🚫 Invalid Reasons:"))
                for k, v in sorted(
                    inv_reasons.items(), key=lambda x: (-int(x[1]), x[0])
                )[:5]:
                    print(VAL(f"  - {k}: {v}"))
            if relaxed_reasons:
                print(KEY("- ✅ Relaxed Reasons:"))
                for k, v in sorted(
                    relaxed_reasons.items(), key=lambda x: (-int(x[1]), x[0])
                )[:5]:
                    print(VAL(f"  - {k}: {v}"))
            inv_samples = val.get("invalid_reason_samples", {})
            if isinstance(inv_samples, dict) and inv_samples:
                print(KEY("- 🔎 Invalid Samples:"))
                for k, urls in inv_samples.items():
                    if isinstance(urls, list) and urls:
                        sl = ", ".join(urls[:3])
                        print(VAL(f"  - {k}: {sl}"))

        # Section: Largest Pages
        if pages:
            try:

                def _page_size_bytes(p):
                    try:
                        return len((getattr(p, "content", "") or "").encode("utf-8"))
                    except Exception:
                        return 0

                def _fmt_bytes(n: int | float) -> str:
                    n = float(n)
                    units = ["B", "KB", "MB", "GB"]
                    i = 0
                    while n >= 1024 and i < len(units) - 1:
                        n /= 1024.0
                        i += 1
                    return f"{n:.2f} {units[i]}"

                largest = sorted(pages, key=_page_size_bytes, reverse=True)[:5]
                if largest:
                    print("\n" + H("📚 Largest Pages"))
                    for i, pg in enumerate(largest, 1):
                        size = _fmt_bytes(_page_size_bytes(pg))
                        title = getattr(pg, "title", "") or pg.url
                        print(VAL(f"  {i}. {title} ({size})"))
            except Exception:
                pass

        # Section: Slowest Pages (by crawl time if available)
        if pages:
            try:

                def _page_crawl_time(p):
                    try:
                        md = getattr(p, "metadata", {}) or {}
                        return float(md.get("crawl_time", 0.0))
                    except Exception:
                        return 0.0

                slowest = sorted(pages, key=_page_crawl_time, reverse=True)[:5]
                if slowest and _page_crawl_time(slowest[0]) > 0:
                    print("\n" + H("⏱️ Slowest Pages"))
                    for i, pg in enumerate(slowest, 1):
                        ct = _page_crawl_time(pg)
                        title = getattr(pg, "title", "") or pg.url
                        print(VAL(f"  {i}. {title} ({ct:.2f}s)"))
            except Exception:
                pass

        # Section: Report Metrics (detailed)
        summ = report.get("summary", {}) or {}
        if summ:
            print("\n" + H("📈 Report Metrics"))
            print(
                KEY("- 🕒 Duration:"),
                VAL(f"{float(summ.get('total_duration', 0.0)):.2f}s"),
            )
            print(KEY("- 📄 Pages Crawled:"), VAL(str(summ.get("pages_crawled", 0))))
            print(KEY("- ❌ Pages Failed:"), VAL(str(summ.get("pages_failed", 0))))
            sr = summ.get("success_rate", 0)
            print(KEY("- ✅ Success Rate:"), VAL(f"{float(sr) * 100:.1f}%"))
            print(
                KEY("- ⚡ Pages/sec:"),
                VAL(f"{float(summ.get('pages_per_second', 0.0)):.2f}"),
            )
            print(KEY("- 📦 Total Content:"), VAL(summ.get("total_content_human", "-")))
            print(
                KEY("- 📏 Avg Page Size:"),
                VAL(summ.get("average_page_size_human", "-")),
            )
            if (qs := summ.get("content_quality_score")) is not None:
                print(KEY("- 🧪 Quality Score:"), VAL(f"{float(qs):.2f}"))

        can = report.get("content_analysis", {}) or {}
        if can:
            print("\n" + H("🧠 Content Analysis"))
            print(KEY("- 📦 Total Content:"), VAL(can.get("total_content_human", "-")))
            print(
                KEY("- 📏 Avg Content Length:"),
                VAL(can.get("average_content_human", "-")),
            )
            cr = can.get("content_range", {}) or {}
            if cr:
                print(
                    KEY("- 📊 Content Range:"),
                    VAL(f"min {cr.get('min', 0)} .. max {cr.get('max', 0)}"),
                )
            print(
                KEY("- 🧩 Hash Placeholders:"),
                VAL(str(can.get("hash_placeholders", 0))),
            )
            print(KEY("- 🔁 Duplicates:"), VAL(str(can.get("duplicates", 0))))
            dups = can.get("duplicate_groups", [])
            if isinstance(dups, list) and dups:
                print(KEY("- 🧬 Top Duplicate Groups:"))
                for i, g in enumerate(dups[:5], 1):
                    first = g.get("first_url", "")
                    cnt = g.get("count", 0)
                    print(VAL(f"  {i}. {first} (x{cnt})"))
                    samples = g.get("sample_urls", []) or []
                    if samples:
                        print(VAL("     ↳ " + ", ".join(samples[:3])))

        # Section: Performance Trend
        trend = report.get("performance_trend", {}) or {}
        if trend:
            print("\n" + H("📉 Performance Trend"))
            print(
                KEY("- Trend:"),
                VAL(str(trend.get("trend", trend.get("status", "n/a")))),
            )
            if trend.get("recent_rate") is not None:
                print(
                    KEY("- Recent rate:"),
                    VAL(f"{float(trend.get('recent_rate', 0.0)):.2f} p/s"),
                )
            if trend.get("overall_rate") is not None:
                print(
                    KEY("- Overall rate:"),
                    VAL(f"{float(trend.get('overall_rate', 0.0)):.2f} p/s"),
                )
            if trend.get("sample_count") is not None:
                print(KEY("- Samples:"), VAL(str(trend.get("sample_count", 0))))

        # Section: System Metrics
        sysm = report.get("system_performance", {})
        if sysm:
            print("\n" + H("🖥️ System"))
            peak_mb = sysm.get("peak_memory_usage_mb")
            avg_cpu = sysm.get("average_cpu_usage")
            proc_cpu = sysm.get("process_cpu_avg")
            peak_conc = sysm.get("concurrent_sessions_peak")
            if peak_mb is not None:
                print(KEY("- 📈 Peak Memory:"), VAL(f"{float(peak_mb):.0f} MB"))
            if avg_cpu is not None:
                print(KEY("- 🧮 Avg CPU:"), VAL(_fmt_pct(avg_cpu)))
            if proc_cpu is not None:
                print(KEY("- 🧑‍💻 Proc CPU:"), VAL(_fmt_pct(proc_cpu)))
        if peak_conc is not None:
            print(KEY("- 🧵 Peak Concurrency:"), VAL(str(peak_conc)))

        # Section: Embeddings (if any)
        emb = report.get("embeddings", {})
        if emb:
            print("\n" + H("🧩 Embeddings"))
            if emb.get("endpoint"):
                print(KEY("- 🌐 Endpoint:"), VAL(str(emb.get("endpoint"))))
            if emb.get("model"):
                print(KEY("- 🧠 Model:"), VAL(str(emb.get("model"))))
            if emb.get("vector_dim"):
                print(KEY("- 📐 Vector Dim:"), VAL(str(emb.get("vector_dim"))))
            if emb.get("pages") is not None:
                print(KEY("- 📄 Pages Embedded:"), VAL(str(emb.get("pages", 0))))
            if emb.get("batches") is not None:
                print(KEY("- 📦 Batches:"), VAL(str(emb.get("batches", 0))))
            if emb.get("batch_size") is not None:
                print(KEY("- 📏 Batch Size:"), VAL(str(emb.get("batch_size", 0))))
            if emb.get("parallel_requests") is not None:
                print(
                    KEY("- 🔀 Parallel Requests:"),
                    VAL(str(emb.get("parallel_requests", 0))),
                )
            if emb.get("avg_batch_latency_ms") is not None:
                print(
                    KEY("- ⏱ Avg Batch Latency:"),
                    VAL(f"{float(emb.get('avg_batch_latency_ms', 0.0)):.1f} ms"),
                )

        # Section: Vector Store (if any)
        vs = report.get("vector_store", {})
        if vs:
            print("\n" + H("🗂 Vector Store"))
            if vs.get("provider"):
                print(KEY("- 🧩 Provider:"), VAL(str(vs.get("provider"))))
            if vs.get("url"):
                print(KEY("- 🌐 URL:"), VAL(str(vs.get("url"))))
            if vs.get("collection"):
                print(KEY("- 📚 Collection:"), VAL(str(vs.get("collection"))))
            if vs.get("points") is not None:
                print(KEY("- 🔢 Points Upserted:"), VAL(str(vs.get("points", 0))))
            if vs.get("batches") is not None:
                print(KEY("- 📦 Batches:"), VAL(str(vs.get("batches", 0))))
            if vs.get("batch_size") is not None:
                print(KEY("- 📏 Batch Size:"), VAL(str(vs.get("batch_size", 0))))
            if vs.get("parallel_requests") is not None:
                print(
                    KEY("- 🔀 Parallel Requests:"),
                    VAL(str(vs.get("parallel_requests", 0))),
                )
            if vs.get("avg_batch_latency_ms") is not None:
                print(
                    KEY("- ⏱ Avg Batch Latency:"),
                    VAL(f"{float(vs.get('avg_batch_latency_ms', 0.0)):.1f} ms"),
                )

        # Section: Errors
        errs = report.get("error_analysis", {})
        if errs:
            print("\n" + H("⚠️ Errors"))
            er = errs.get("error_rate")
            total_err = errs.get("total_errors")
            if er is not None:
                print(
                    KEY("- 📉 Error Rate:"), BAD(_fmt_pct(er * 100 if er < 1 else er))
                )
            if total_err is not None:
                print(KEY("- ❌ Total Errors:"), BAD(str(total_err)))
            breakdown = errs.get("error_breakdown", {})
            if breakdown:
                print(KEY("- 🗂️ Breakdown:"))
                for k, v in sorted(breakdown.items(), key=lambda x: (-int(x[1]), x[0]))[
                    :5
                ]:
                    print(BAD(f"  - {k}: {v}"))
            samples = errs.get("error_samples", {})
            if isinstance(samples, dict) and samples:
                print(KEY("- 🧪 Samples by Reason:"))
                for k, urls in samples.items():
                    if isinstance(urls, list) and urls:
                        sl = ", ".join(urls[:3])
                        print(BAD(f"  - {k}: {sl}"))

        # Section: Recommendations
        recs = report.get("recommendations", [])
        if recs:
            print("\n" + H("💡 Recommendations"))
            for r in recs[:5]:
                print(VAL(f"- {r}"))

        # Outputs section
        out_lines = []
        if args.output_html:
            out_lines.append(("📄 HTML", args.output_html))
        if args.output_ndjson:
            out_lines.append(("🧾 NDJSON", args.output_ndjson))
        if args.report_json:
            out_lines.append(("📊 Report JSON", args.report_json))
        if out_lines:
            print("\n" + H("📝 Outputs"))
            for label, path in out_lines:
                print(KEY(f"- {label}:"), VAL(path))

        # Optional: write JSON report
        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return 0
    finally:
        await strat.close()


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
