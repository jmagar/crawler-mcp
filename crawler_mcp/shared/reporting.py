"""Terminal reporting helpers for crawl summaries."""

from __future__ import annotations

import os
import sys
from typing import Any


def print_enhanced_report(
    args, headers: dict[str, Any], report: dict[str, Any], strat
) -> dict[str, Any]:
    """Pretty-print the enhanced crawl report and return possibly augmented report.

    This prints a human-friendly summary (with color when TTY) and, when the
    extraction method is the GitHub PR fast-path, prints a PR-specific section
    and attaches a compact github_pr_report into the report dict.
    """

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

    def _supports_color() -> bool:
        try:
            return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        except Exception:
            return False

    _COLOR = _supports_color()

    def _sty(s: str, code: str) -> str:
        return f"\033[{code}m{s}\033[0m" if _COLOR else s

    def H(s: str) -> str:  # noqa: N802
        return _sty(s, "1;36")

    def KEY(s: str) -> str:  # noqa: N802
        return _sty(s, "90")

    def VAL(s: str) -> str:  # noqa: N802
        return _sty(s, "36")

    def GOOD(s: str) -> str:  # noqa: N802
        return _sty(s, "32")

    def BAD(s: str) -> str:  # noqa: N802
        return _sty(s, "31")

    def WARN(s: str) -> str:  # noqa: N802
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

    line = "".ljust(70, "=")
    print(_sty(line, "90"))
    print(H(" 🚀 Optimized Crawl Report "))
    print(_sty(line, "90"))

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
        BAD(failed_count) if failed_count and failed_count != "0" else KEY(failed_count)
    )
    print(KEY("- ❗ Failed URLs:"), fc)

    print("\n" + H("🔗 Content & Links"))
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
            for link in page_links:
                if not link:
                    continue
                unique_link_set.add(link)
                if _urlparse:
                    try:
                        host = _urlparse(link).netloc
                        if host:
                            host_counts[host] = host_counts.get(host, 0) + 1
                    except Exception:
                        pass
        except Exception:
            continue
    unique_links_count = len(unique_link_set)
    print(
        KEY("- 🔗 Unique Links:"), VAL(_fmt_num(str(unique_links_count or total_links)))
    )
    print(KEY("- 🔁 Link Instances:"), VAL(_fmt_num(str(total_link_instances))))
    if sample_links:
        links_list = [s for s in sample_links.split(",") if s][:5]
        if links_list:
            print(KEY("- 🔎 Sample Links:"))
            for i, link in enumerate(links_list, 1):
                print(VAL(f"  🔹 {i}. {link}"))
    if host_counts:
        print(KEY("- 🌐 Top Link Hosts:"))
        for i, (host, cnt) in enumerate(
            sorted(host_counts.items(), key=lambda x: (-x[1], x[0]))[:5], 1
        ):
            print(VAL(f"  {i}. {host} ({cnt})"))

    # Failures
    if failed_count and failed_count != "0":
        print("\n" + H("💥 Failures"))
        if failed_sample:
            for i, link in enumerate([s for s in failed_sample.split(",") if s][:5], 1):
                print(BAD(f"  🔻 {i}. {link}"))

    # Validation & Quality
    val = report.get("validation_summary", {}) or {}
    relaxed_total = val.get("relaxed_acceptances_total") or val.get(
        "relaxed_acceptances"
    )
    inv_reasons = val.get("invalid_reasons", {}) or val.get("invalid_reason_counts", {})
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
            for k, v in sorted(inv_reasons.items(), key=lambda x: (-int(x[1]), x[0]))[
                :5
            ]:
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

    # Largest pages
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

    # Slowest pages
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

    # Summary metrics
    summ = report.get("summary", {}) or {}
    if summ:
        print("\n" + H("📈 Report Metrics"))
        print(
            KEY("- 🕒 Duration:"), VAL(f"{float(summ.get('total_duration', 0.0)):.2f}s")
        )
        print(KEY("- 📄 Pages Crawled:"), VAL(str(summ.get("pages_crawled", 0))))
        print(KEY("- ❌ Pages Failed:"), VAL(str(summ.get("pages_failed", 0))))
        sr = summ.get("success_rate", 0)
        print(
            KEY("- ✅ Success Rate:"), VAL(_fmt_pct(sr * 100 if sr and sr < 1 else sr))
        )
        print(
            KEY("- ⚡ Pages/sec:"),
            VAL(f"{float(summ.get('pages_per_second', 0.0)):.2f}"),
        )
        print(
            KEY("- 📦 Total Content:"), VAL(str(summ.get("total_content_human", "-")))
        )
        print(
            KEY("- 📏 Avg Page Size:"), VAL(str(summ.get("avg_page_size_human", "-")))
        )

    # System performance
    sysm = report.get("system_performance", {})
    peak_conc = None
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

    # Embeddings
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

    # Vector store
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
                KEY("- 🔀 Parallel Requests:"), VAL(str(vs.get("parallel_requests", 0)))
            )
        if vs.get("avg_batch_latency_ms") is not None:
            print(
                KEY("- ⏱ Avg Batch Latency:"),
                VAL(f"{float(vs.get('avg_batch_latency_ms', 0.0)):.1f} ms"),
            )

    # Errors
    errs = report.get("error_analysis", {})
    if errs:
        print("\n" + H("⚠️ Errors"))
        er = errs.get("error_rate")
        total_err = errs.get("total_errors")
        if er is not None:
            print(KEY("- 📉 Error Rate:"), BAD(_fmt_pct(er * 100 if er < 1 else er)))
        if total_err is not None:
            print(KEY("- ❌ Total Errors:"), BAD(str(total_err)))
        breakdown = errs.get("error_breakdown", {})
        if breakdown:
            print(KEY("- 🗂️ Breakdown:"))
            for k, v in sorted(breakdown.items(), key=lambda x: (-int(x[1]), x[0]))[:5]:
                print(BAD(f"  - {k}: {v}"))
        samples = errs.get("error_samples", {})
        if isinstance(samples, dict) and samples:
            print(KEY("- 🧪 Samples by Reason:"))
            for k, urls in samples.items():
                if isinstance(urls, list) and urls:
                    sl = ", ".join(urls[:3])
                    print(BAD(f"  - {k}: {sl}"))

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        print("\n" + H("💡 Recommendations"))
        for r in recs[:5]:
            print(VAL(f"- {r}"))

    # Outputs - now handled by OutputManager, show output directory
    output_dir = getattr(args, "output_dir", "./.crawl4ai")
    if not getattr(args, "skip_output", False):
        print("\n" + H("📝 Outputs"))
        print(KEY("- 📂 Output Directory:"), VAL(output_dir))
        print(KEY("- 📄 HTML:"), VAL("combined.html"))
        print(KEY("- 🧾 NDJSON:"), VAL("pages.ndjson"))
        print(KEY("- 📊 Report JSON:"), VAL("report.json"))

    # GitHub PR report
    try:
        if headers.get("X-Extraction-Method", "") == "github_pr_api":
            pr_rep = strat.get_pr_report() if hasattr(strat, "get_pr_report") else None
            if pr_rep:
                print("\n" + H("🧾 GitHub PR Report"))
                print(
                    KEY("- Repo:"),
                    VAL(f"{pr_rep.get('owner', '')}/{pr_rep.get('repo', '')}"),
                )
                print(
                    KEY("- PR:"),
                    VAL(f"#{pr_rep.get('pr_number', '')} {pr_rep.get('title', '')}"),
                )
                print(
                    KEY("- State:"),
                    VAL("merged" if pr_rep.get("merged") else pr_rep.get("state", "")),
                )
                print(KEY("- Author:"), VAL(str(pr_rep.get("author", ""))))
                print(
                    KEY("- Reviews:"),
                    VAL(str(pr_rep.get("reviews_total", 0))),
                    KEY(str(pr_rep.get("review_states", {}))),
                )
                print(
                    KEY("- Review Comments:"),
                    VAL(str(pr_rep.get("review_comments_total", 0))),
                )
                print(
                    KEY("- Conversation Comments:"),
                    VAL(str(pr_rep.get("conversation_comments_total", 0))),
                )
                print(
                    KEY("- Participants:"),
                    VAL(", ".join(pr_rep.get("participants", []))),
                )
                files = pr_rep.get("files", []) or []
                if files:
                    try:
                        files_sorted = sorted(
                            files, key=lambda x: int(x.get("comments", 0)), reverse=True
                        )
                    except Exception:
                        files_sorted = files
                    top = files_sorted[:5]
                    if top:
                        print(KEY("- Files (top by comments):"))
                        for it in top:
                            path = it.get("path", "")
                            comments = it.get("comments", 0)
                            changes = it.get("changes", 0)
                            print(
                                "   ",
                                VAL(path),
                                KEY(f"comments={comments}, changes={changes}"),
                            )
                try:
                    if isinstance(report, dict):
                        report["github_pr_report"] = pr_rep
                except Exception:
                    pass
    except Exception:
        pass

    return report
