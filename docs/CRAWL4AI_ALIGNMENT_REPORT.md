# Crawl4AI Pattern Alignment Report

Date: 2025-09-06

This report documents how the current implementation aligns with Crawl4AI’s recommended patterns, identifies divergences, and provides actionable recommendations. Sources include code review of key modules in this repo and targeted queries against the crawl4ai repository.

## Scope Reviewed

- `crawler_mcp/crawl_core/adaptive_dispatcher.py`
- `crawler_mcp/crawl_core/batch_utils.py`
- `crawler_mcp/crawl_core/parallel_engine.py`
- `crawler_mcp/crawl_core/strategy.py`


## Current Implementation Overview

- Strategy: `OptimizedCrawlerStrategy` extends Crawl4AI’s `AsyncCrawlerStrategy` and orchestrates the full pipeline: discovery → parallel crawl (arun_many) → validation → conversion → optional embeddings/Qdrant → reporting.
- Parallel engine: `ParallelEngine` performs high‑throughput crawling via `AsyncWebCrawler.arun_many`, supports streaming and batch modes, content validation (hash/quality heuristics), optional JS retry, placeholder retry pass, and request interception for resource blocking.
- Concurrency tuning: `ConcurrencyTuner` adjusts `MemoryAdaptiveDispatcher.max_session_permit` at runtime using `PerformanceMonitor` metrics (CPU, error‑rate).
- Settings: `CrawlerSettings` centralizes discovery, concurrency, extraction, embeddings (TEI), and Qdrant settings, with env overrides via environment variables.
- Batching utils: Greedy length‑aware batching for TEI payload shaping and GPU utilization (`pack_texts_into_batches` et al.).

## Crawl4AI Patterns: What “Good” Looks Like

- Async lifecycle and strategy
  - Use `AsyncWebCrawler` with `async with` lifecycle. Strategies are orchestrators around it.
  - `arun_many` supports both list (batch) and async generator (stream). Toggle via `CrawlerRunConfig.stream=True`.
- Dispatcher and concurrency
  - `MemoryAdaptiveDispatcher(memory_threshold_percent, max_session_permit, check_interval, ...)` is the default for `arun_many`. It protects against OOM and provides upper bound concurrency via `max_session_permit`.
  - Monitoring via optional `CrawlerMonitor`; per‑task `dispatch_result` may be attached on results.
- CrawlerRunConfig stability
  - Stable, recommended fields: `page_timeout`, `delay_before_return_html`, `wait_for` (css/js), `word_count_threshold`, `excluded_tags`, `css_selector`, `process_iframes`, `cache_mode`, `stream`.
  - Prefer `wait_for` and `js_code` to handle JS‑rendered content. JavaScript execution is on by default when using a browser strategy.
- Resource reduction
  - Prefer built‑in Text‑Only mode (via Playwright crawler strategy config) to block heavy resources; alternatively use lightweight HTTP strategy where JS rendering is not needed.
- Content fields on results
  - `CrawlResult` exposes `html`, `cleaned_html`, and `markdown` object (`raw_markdown`, `fit_markdown`, `markdown_with_citations`) plus `extracted_content` for structured extractors.

References: crawl4ai docs for `arun_many`, async crawler lifecycle, dispatcher usage, parameters, and content selection.

## Cross‑Reference: Our Implementation vs Crawl4AI Patterns

- `arun_many` usage
  - We correctly handle both modes: detect async generator via `hasattr(gen, "__aiter__")` and iterate; otherwise treat as list. We also support `stream=True`.
  - We pass an optional dispatcher and keep a monitor. This matches patterns.
- Dispatcher/concurrency
  - We instantiate/passthrough external dispatcher and add a `ConcurrencyTuner` that modifies `dispatcher.max_session_permit` at runtime. Crawl4AI sets this at construction; direct runtime changes are not documented but the attribute exists publicly. Considered “works but unofficial” — keep guarded and best‑effort.
- Resource blocking
  - We implement manual Playwright `route` interception to abort images/media/fonts/analytics. Crawl4AI recommends Text‑Only mode for this pattern. Our manual handler is acceptable but brittle across strategy internals. Consider switching to text‑mode at the crawler strategy level when possible.
- CrawlerRunConfig fields
  - We set stable fields (`page_timeout`, `delay_before_return_html`, `excluded_tags`, etc.) and also reference `enable_javascript` and `wait_for_js_rendering`. Crawl4AI docs do not define these flags on `CrawlerRunConfig` (JavaScript is enabled by default; use `wait_for`/`js_code`). Using these attributes may be version‑fragile. Prefer `wait_for` and content‑ready conditions; avoid relying on non‑documented fields.
- Content validation
  - We add extra heuristics: hash‑like detection, word/length checks, sentence heuristics, relaxed acceptance for docs patterns. Crawl4AI does not include a specific “hash placeholder” detector; our layer is fine and complementary to built‑in content selection (`word_count_threshold`, filters).
- Placeholder/JS retry and fallback discovery
  - We implement bounded retry for placeholder pages (optionally enabling JS, increasing timeout). Not an official pattern but congruent with operational recovery.
  - Fallback discovery from in‑page links when sitemap yield is low, with locale filtering and JS‑aware single fetch. This is a repo‑specific enhancement and aligns with the goal of improving coverage; not part of base Crawl4AI.
- Stats/monitoring
  - We compute session stats and record success/failure/quality metrics via `PerformanceMonitor`; this complements Crawl4AI’s optional `CrawlerMonitor`.

## Notable Divergences and Risks

- Runtime tuning of `max_session_permit`
  - While the attribute exists, Crawl4AI does not explicitly document runtime mutation during active dispatch. Keep changes infrequent, protected by try/except, and bound by safe min/max. Consider exposing a sanctioned callback/hook if upstream provides one.
- Manual Playwright routing vs Text‑Only mode
  - Our manual route interception works but is strategy‑internal and may break if underlying page/context properties change. Prefer strategy configuration for text mode to ensure resource blocking is maintained across versions.
- Non‑documented CrawlerRunConfig fields
  - `enable_javascript`, `wait_for_js_rendering` are not in documented `CrawlerRunConfig`. For portability, migrate to `wait_for` (css/js conditions) and `delay_before_return_html`. Treat current usage as best‑effort and optional.
- Result field assumptions
  - Accessing `markdown.fit_markdown` and `markdown.raw_markdown` is aligned with docs, but always guard with hasattr checks (already present) since formatting objects can evolve.

## Actionable Recommendations

1. Prefer text‑mode configuration over manual route interception
   - Move resource blocking to crawler strategy settings (Text‑Only mode) when JS is not required. Keep the route handler as a fallback behind a feature flag.

2. Replace non‑documented run‑config flags with stable ones
   - Migrate `enable_javascript`/`wait_for_js_rendering` usage toward:
     - `wait_for="css:<selector>"` or `wait_for="js:(...)"` for readiness.
     - `delay_before_return_html` for a minimal buffer.
     - Use `js_code` for page interaction (scroll/click) when needed.

3. Make concurrency tuning opt‑in and bounded
   - Keep `ConcurrencyTuner` optional, with config gating, and log when overridden. Limit adjustment frequency and step size; never exceed initial `max_session_permit` unless explicitly allowed by config.

4. Embrace `CrawlerMonitor` when available
   - If feasible, integrate Crawl4AI's `CrawlerMonitor` for live visibility and expose those metrics alongside our `PerformanceMonitor` report.

5. Harden validation feature flags
   - Allow toggling validation strictness per domain/route (e.g., docs vs apps), and expose `doc_relax_validation_patterns` in config (already present) with clear defaults.

6. Align config cloning to documented stable fields
   - In `_clone_config`, prefer copying only documented fields, and add a safe pathway for `wait_for`/`js_code`. Keep `excluded_tags`, `word_count_threshold`, `page_timeout`, `delay_before_return_html`, `cache_mode`, `exclude_external_links`, `process_iframes`, `remove_overlay_elements`, `only_text`.

## Q&A Highlights from crawl4ai Repository

- `arun_many` return type: list vs async generator is controlled by `CrawlerRunConfig.stream`.
- Dispatcher usage: `MemoryAdaptiveDispatcher(memory_threshold_percent, max_session_permit, ...)` is preferred for multi‑URL crawling. `max_session_permit` exists as an attribute; docs show setting it on construction. Runtime mutation is not explicitly covered.
- Content handling: `CrawlResult` provides `html`, `cleaned_html`, and `markdown` flavors (`raw_markdown`, `fit_markdown`, etc.).
- JavaScript handling: No `enable_javascript`/`wait_for_js_rendering` flags on `CrawlerRunConfig`. Use `wait_for` and `js_code`; JS is enabled by default in browser strategies.
- Resource blocking: Prefer Text‑Only mode or HTTP strategy for lightweight crawls; manual routing is not the recommended primary pattern.

## Proposed Follow‑Ups

- Add a config switch `use_text_mode_resource_blocking` to replace manual routing when appropriate.
- Introduce `wait_for` integration points in factories to standardize readiness conditions per site class.
- Optionally integrate Crawl4AI’s `CrawlerMonitor` and surface its aggregated stats in our `PerformanceMonitor` final report.
- Add tests covering both streaming and batch `arun_many` flows to ensure compatibility.

---

Prepared by: Codex CLI agent
