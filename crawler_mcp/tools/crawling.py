"""
Optimized crawling tools: `scrape` (single page) and `crawl` (unified).

Both tools live under the optimized namespace and use only optimized/core
components plus shared models. RAG ingestion is auto-enabled when TEI and
Qdrant are configured, unless explicitly overridden per-call.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from crawler_mcp.crawl_core.strategy import CrawlOrchestrator
from crawler_mcp.middleware.progress import progress_middleware
from crawler_mcp.models.crawl import PageContent
from crawler_mcp.optimized_config import OptimizedConfig
from crawler_mcp.utils.output_manager import OutputManager

_HASH32_40_64_RE = re.compile(
    r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|[A-Za-z0-9]{32})$"
)

MAX_CLEAN_LEN = 50_000


def _strip_html_basic(html: str) -> str:
    # Very lightweight HTML tag stripper as a last resort (no external deps)
    try:
        # remove scripts/styles
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
        # remove tags
        text = re.sub(r"<[^>]+>", " ", html)
        # collapse whitespace
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return html


def _get_clean_content(resp: Any, pages: list[PageContent]) -> str:
    """Choose the best cleaned content from response/pages with safe fallbacks."""
    # 1) Optimized response extracted content
    try:
        ec = getattr(resp, "extracted_content", "") or ""
        if ec and not _is_placeholder_text(ec):
            return ec
    except Exception:
        pass

    # Helper to pull markdown variants from a page-like object
    def page_markdown(page: Any) -> str:
        try:
            md = getattr(page, "markdown", None)
            if md:
                return md
            # Some conversions may keep fit/raw attributes inside markdown-like object
            for attr in ("fit_markdown", "raw_markdown"):
                val = getattr(page, attr, None)
                if val:
                    return str(val)
        except Exception:
            pass
        return ""

    # 2) First page markdown
    if pages:
        md = page_markdown(pages[0])
        if md and not _is_placeholder_text(md):
            return md

    # 3) First page plaintext content
    if pages:
        try:
            txt = getattr(pages[0], "content", "") or ""
            if txt and not _is_placeholder_text(txt):
                return txt
        except Exception:
            pass

    # 4) First page HTML stripped
    if pages:
        try:
            html = getattr(pages[0], "html", "") or ""
            if html:
                stripped = _strip_html_basic(html)
                if stripped and not _is_placeholder_text(stripped):
                    return stripped
        except Exception:
            pass

    return ""


def _is_placeholder_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    # very short overall
    if len(t) < 16:
        return True
    # check first lines for hash-like placeholders or hash walls
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    first = lines[0]
    if _HASH32_40_64_RE.match(first):
        return True
    if len(first) >= 8 and set(first) == {"#"}:
        return True
    if t.lower().startswith("hash:"):
        return True
    # minimal tokens
    return len(re.findall(r"\w+", t)) < 8


def _env(var: str) -> str | None:
    v = os.getenv(var)
    return v if v and v.strip() else None


def _should_ingest_rag(rag_ingest: bool | None) -> bool:
    """Tri-state RAG toggle: True/False/None(auto)."""
    if rag_ingest is True:
        return True
    if rag_ingest is False:
        return False
    # Auto: on when both endpoints exist
    tei = _env("OPTIMIZED_CRAWLER_TEI_ENDPOINT") or _env("TEI_URL")
    qdrant = _env("OPTIMIZED_CRAWLER_QDRANT_URL") or _env("QDRANT_URL")
    return bool(tei and qdrant)


def _apply_rag_to_config(cfg: OptimizedConfig, enable: bool) -> None:
    """Apply RAG auto/override to OptimizedConfig (embeddings + qdrant)."""
    if enable:
        # Prefer optimized-prefixed envs; fall back to standard ones
        tei = _env("OPTIMIZED_CRAWLER_TEI_ENDPOINT") or _env("TEI_URL")
        qurl = _env("OPTIMIZED_CRAWLER_QDRANT_URL") or _env("QDRANT_URL")
        qcol = _env("OPTIMIZED_CRAWLER_QDRANT_COLLECTION") or _env("QDRANT_COLLECTION")
        if not (tei and qurl):
            raise ToolError(
                "RAG ingestion requested but TEI and/or Qdrant endpoints are missing"
            )
        cfg.enable_embeddings = True
        cfg.enable_qdrant = True
        cfg.tei_endpoint = tei
        cfg.qdrant_url = qurl
        if qcol:
            cfg.qdrant_collection = qcol
    else:
        cfg.enable_embeddings = False
        cfg.enable_qdrant = False


def _detect_target(
    target: str,
) -> Literal["website", "github_pr", "repository", "directory"]:
    if re.match(r"^https?://", target, re.IGNORECASE):
        # PR fast-path
        if re.match(
            r"^https://github\.com/[^/]+/[^/]+/pull/\d+", target, re.IGNORECASE
        ):
            return "github_pr"
        return "website"
    if os.path.isdir(target):
        return "directory"
    if target.endswith(".git") or target.startswith("git@") or "github.com" in target:
        return "repository"
    raise ToolError(f"Unsupported or undetected target type: {target}")


def _read_text_file(path: Path, size_limit: int) -> str:
    try:
        if path.stat().st_size > size_limit:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


async def _crawl_directory(
    ctx: Context,
    root: Path,
    include: list[str] | None,
    exclude: list[str] | None,
    max_files: int,
    file_size_limit_bytes: int,
) -> list[PageContent]:
    await ctx.info(f"Scanning directory: {root}")
    patterns = include or ["**/*.md", "**/*.txt", "**/*.html", "**/*.py"]
    seen: set[Path] = set()
    pages: list[PageContent] = []
    for pat in patterns:
        for p in root.rglob(pat):
            if not p.is_file():
                continue
            if exclude and any(p.match(ex) for ex in exclude):
                continue
            if p in seen:
                continue
            seen.add(p)
            content = _read_text_file(p, file_size_limit_bytes)
            if not content:
                continue
            # Filter placeholder-like files
            if _is_placeholder_text(content):
                continue
            pages.append(
                PageContent(
                    url=f"file://{p}",
                    title=p.name,
                    content=content,
                    markdown=content
                    if p.suffix.lower() in {".md", ".markdown"}
                    else None,
                    html=None,
                    links=[],
                    images=[],
                    metadata={"path": str(p)},
                )
            )
            if len(pages) >= max_files:
                break
        if len(pages) >= max_files:
            break
    return pages


async def _crawl_repository(
    ctx: Context,
    repo_url: str,
    include: list[str] | None,
    exclude: list[str] | None,
    max_files: int,
    file_size_limit_bytes: int,
) -> tuple[list[PageContent], Path]:
    await ctx.info(f"Cloning repository: {repo_url}")
    tmpdir = Path(tempfile.mkdtemp(prefix="repo_crawl_"))
    try:
        try:
            from git import Repo  # type: ignore
        except Exception as e:
            raise ToolError(
                "GitPython is required to crawl repositories; ensure dependency is installed"
            ) from e
        Repo.clone_from(repo_url, tmpdir, depth=1, no_single_branch=True)
        pages = await _crawl_directory(
            ctx,
            tmpdir,
            include=include,
            exclude=exclude,
            max_files=max_files,
            file_size_limit_bytes=file_size_limit_bytes,
        )
        return pages, tmpdir
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise ToolError(f"Failed to clone or scan repository: {e}") from e


def register_crawling_tools(mcp: FastMCP) -> None:
    """Register `scrape` and `crawl` tools on the provided FastMCP instance."""

    @mcp.tool
    async def scrape(
        ctx: Context,
        url: str,
        screenshot: bool = False,  # reserved for future screenshot support
        wait_for: str | None = None,
        css_selector: str | None = None,
        javascript: bool | None = None,
        timeout_ms: int = 30000,
        rag_ingest: bool | None = None,
    ) -> dict[str, Any]:
        """Single Page Intelligence: extract one web page with optimized pipeline."""
        if not re.match(r"^https?://", url, re.IGNORECASE):
            raise ToolError("Invalid URL; must start with http:// or https://")

        tracker = progress_middleware.create_tracker(f"scrape_{hash(url)}")
        try:
            await ctx.info(f"Starting scrape of: {url}")
            await ctx.report_progress(progress=5, total=100)

            cfg = OptimizedConfig.from_env()
            if javascript is not None:
                cfg.browser_mode = "full" if bool(javascript) else "text"
            cfg.page_timeout = max(1000, int(timeout_ms))

            rag_enabled = _should_ingest_rag(rag_ingest)
            _apply_rag_to_config(cfg, rag_enabled)

            output = OutputManager(config=cfg)
            domain = output.sanitize_domain(url)
            output.rotate_crawl_backup(domain)

            await ctx.report_progress(progress=20, total=100)

            # Generate unique session ID for this crawl
            session_id = str(uuid.uuid4())
            await ctx.info(f"Crawl session ID: {session_id}")

            strategy = CrawlOrchestrator(cfg)
            await ctx.info("Calling strategy.crawl with max_urls=1")
            try:
                resp = await strategy.crawl(
                    url,
                    max_urls=1,
                    max_concurrent=max(1, cfg.max_concurrent_crawls),
                )
                await ctx.info(
                    f"Strategy returned success: {getattr(resp, 'success', 'unknown')}"
                )
            except Exception as e:
                await ctx.info(
                    f"Strategy.crawl threw exception: {type(e).__name__}: {e}"
                )
                # Re-raise to let normal error handling proceed
                raise

            pages = getattr(strategy, "get_last_pages", lambda: [])()
            # Compute cleaned content for return
            clean = _get_clean_content(resp, pages)

            paths = output.get_crawl_output_paths(url, session_id)
            output.save_crawl_outputs(
                domain=domain,
                html=getattr(resp, "html", "") or None,
                pages=pages,
                report=getattr(resp, "metadata", {}) or {},
                session_id=session_id,
            )
            output.create_latest_symlink(domain, session_id)
            output.update_index(
                domain,
                {
                    "url": url,
                    "pages": len(pages),
                    "rag_enabled": rag_enabled,
                },
                session_id,
            )
            clean_truncated = False
            if len(clean) > MAX_CLEAN_LEN:
                clean = clean[:MAX_CLEAN_LEN]
                clean_truncated = True

            content_preview = (
                clean or getattr(resp, "extracted_content", "") or ""
            ).strip()[:500]
            title = pages[0].title if pages else None
            word_count = pages[0].word_count if pages else 0

            await ctx.report_progress(progress=100, total=100)

            # Separate network success from content extraction success
            network_success = bool(getattr(resp, "success", True))
            content_extracted = bool(clean)

            # Overall success if network succeeded, regardless of content extraction
            overall_success = network_success

            # Add diagnostic information
            diagnostics = {
                "network_success": network_success,
                "content_extracted": content_extracted,
                "pages_found": len(pages),
                "retry_js_attempted": False,
            }

            # If network succeeded but no content, add diagnosis
            if network_success and not content_extracted:
                await ctx.info(
                    f"Network request succeeded but no content extracted from {url}"
                )
                diagnostics["extraction_issue"] = "content_empty_after_processing"

            # If network failed, add more details
            if not network_success:
                await ctx.info(f"Network request failed for {url}")
                diagnostics["network_issue"] = "http_request_failed"

            return {
                "success": overall_success,
                "url": url,
                "title": title,
                "word_count": word_count,
                "content_preview": content_preview,
                "outputs": {
                    "html_path": str(paths["html"]),
                    "ndjson_path": str(paths["ndjson"]),
                    "report_path": str(paths["report"]),
                },
                "rag": {"enabled": rag_enabled},
                "clean_content": clean,
                "clean_truncated": clean_truncated,
                "retry_js": False,
                "diagnostics": diagnostics,
            }
        finally:
            progress_middleware.remove_tracker(tracker.operation_id)

    @mcp.tool
    async def crawl(
        ctx: Context,
        target: str,
        limit: int = 100,
        depth: int = 2,
        max_concurrent: int | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        javascript: bool | None = None,
        screenshot_samples: int = 0,  # reserved for future
        timeout_ms: int = 30000,
        rag_ingest: bool | None = None,
    ) -> dict[str, Any]:
        """Unified Smart Crawling: website, directory, repository, or GitHub PR."""
        kind = _detect_target(target)
        tracker = progress_middleware.create_tracker(f"crawl_{hash(target)}")
        try:
            await ctx.report_progress(progress=5, total=100)

            cfg = OptimizedConfig.from_env()
            cfg.page_timeout = max(1000, int(timeout_ms))
            if max_concurrent is not None:
                cfg.max_concurrent_crawls = max(1, int(max_concurrent))
            if javascript is not None:
                cfg.browser_mode = "full" if bool(javascript) else "text"

            rag_enabled = _should_ingest_rag(rag_ingest)
            _apply_rag_to_config(cfg, rag_enabled)

            output = OutputManager(config=cfg)

            await ctx.info("Preparing crawl...")
            await ctx.report_progress(progress=15, total=100)

            # Generate unique session ID for this crawl
            session_id = str(uuid.uuid4())
            await ctx.info(f"Crawl session ID: {session_id}")

            docs_preview: list[dict[str, Any]] = []
            paths: dict[str, Path] | None = None
            success = True
            pages_total = 0
            pages_failed = 0
            duration_s = 0.0
            domain_for_index = ""

            if kind in ("website", "github_pr"):
                url = target
                domain_for_index = output.sanitize_domain(url)
                output.rotate_crawl_backup(domain_for_index)

                strategy = CrawlOrchestrator(cfg)
                resp = await strategy.crawl(
                    url,
                    max_urls=max(1, int(limit)),
                    max_concurrent=max(1, cfg.max_concurrent_crawls),
                )
                pages = getattr(strategy, "get_last_pages", lambda: [])()
                pages_total = len(pages)
                # Try to derive failures from headers if present
                try:
                    failed_ct = int(
                        resp.response_headers.get("X-Failed-URLs-Count", "0")
                    )
                except Exception:
                    failed_ct = 0
                pages_failed = failed_ct
                duration_s = (
                    float(
                        getattr(getattr(resp, "metadata", {}), "get", lambda *_: 0)(
                            "duration_seconds", 0
                        )
                    )
                    or 0.0
                )

                paths = output.get_crawl_output_paths(url, session_id)
                output.save_crawl_outputs(
                    domain=domain_for_index,
                    html=getattr(resp, "html", "") or None,
                    pages=pages,
                    report=getattr(resp, "metadata", {}) or {},
                    session_id=session_id,
                )
                output.create_latest_symlink(domain_for_index, session_id)
                docs_preview = [
                    {
                        "url": p.url,
                        "title": p.title,
                        "words": p.word_count,
                    }
                    for p in pages[:10]
                ]

            elif kind == "directory":
                root = Path(target).expanduser().resolve()
                if not root.exists() or not root.is_dir():
                    raise ToolError(f"Directory does not exist: {root}")
                pages = await _crawl_directory(
                    ctx,
                    root=root,
                    include=include_patterns,
                    exclude=exclude_patterns,
                    max_files=int(limit),
                    file_size_limit_bytes=2_000_000,
                )
                pages_total = len(pages)
                pages_failed = 0
                domain_for_index = f"dir_{root.name}".lower()
                paths = output.get_crawl_output_paths(
                    f"https://{domain_for_index}", session_id
                )
                output.save_crawl_outputs(
                    domain=domain_for_index,
                    html=None,
                    pages=pages,
                    report={"kind": "directory", "root": str(root)},
                    session_id=session_id,
                )
                output.create_latest_symlink(domain_for_index, session_id)
                docs_preview = [
                    {"url": p.url, "title": p.title, "words": p.word_count}
                    for p in pages[:10]
                ]

            elif kind == "repository":
                pages: list[PageContent]
                tmp: Path
                pages, tmp = await _crawl_repository(
                    ctx,
                    repo_url=target,
                    include=include_patterns,
                    exclude=exclude_patterns,
                    max_files=int(limit),
                    file_size_limit_bytes=2_000_000,
                )
                try:
                    repo_name = tmp.name
                except Exception:
                    repo_name = "repo"
                pages_total = len(pages)
                pages_failed = 0
                domain_for_index = f"repo_{repo_name}".lower()
                paths = output.get_crawl_output_paths(
                    f"https://{domain_for_index}", session_id
                )
                output.save_crawl_outputs(
                    domain=domain_for_index,
                    html=None,
                    pages=pages,
                    report={"kind": "repository", "repo": target},
                    session_id=session_id,
                )
                output.create_latest_symlink(domain_for_index, session_id)
                docs_preview = [
                    {"url": p.url, "title": p.title, "words": p.word_count}
                    for p in pages[:10]
                ]
                # Cleanup
                shutil.rmtree(tmp, ignore_errors=True)

            else:  # pragma: no cover - exhaustiveness
                raise ToolError(f"Unsupported target: {target}")

            await ctx.info("Finalizing outputs...")
            await ctx.report_progress(progress=95, total=100)

            # Update index
            if domain_for_index:
                output.update_index(
                    domain_for_index,
                    {
                        "target": target,
                        "kind": kind,
                        "pages": pages_total,
                        "rag_enabled": rag_enabled,
                    },
                    session_id,
                )

            await ctx.report_progress(progress=100, total=100)

            return {
                "success": success,
                "kind": kind,
                "target": target,
                "stats": {
                    "processed": pages_total,
                    "failed": pages_failed,
                    "duration_s": duration_s,
                },
                "outputs": None
                if not paths
                else {
                    "manifest_path": str(paths.get("report", "")),
                    "content_dir": str(Path(paths["ndjson"]).parent),
                    "ndjson_path": str(paths.get("ndjson", "")),
                    "html_path": str(paths.get("html", "")),
                },
                "docs_preview": docs_preview,
                "warnings": [],
                "rag": {"enabled": rag_enabled},
            }
        finally:
            progress_middleware.remove_tracker(tracker.operation_id)
