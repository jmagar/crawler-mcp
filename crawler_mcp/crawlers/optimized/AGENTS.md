# Repository Guidelines

## Project Structure & Module Organization
- `crawler_mcp/`: core package (server, config, tools, middleware, services, webhook).
- `tests/`: pytest suite with asyncio; `unit`, `integration`, `slow`, `requires_services` markers.
- `scripts/`: batch/manual extraction helpers. `docs/`, `examples/`, `data/`, `logs/` for assets/outputs.
- Entry points: `crawler_mcp/server.py` (MCP server), `crawler_mcp/webhook/server.py` (webhook).
- MCP tools live in `crawler_mcp/tools/` and are registered in `crawler_mcp/server.py`.

## Build, Test, and Development Commands
- Install (editable): `uv sync` or `pip install -e .`
- Run server (dev): `fastmcp dev crawler_mcp/server.py`
- Run server (direct): `uv run python -m crawler_mcp.server` or `uv run crawler-mcp`
- Webhook (direct): `uv run python -m crawler_mcp.webhook.server` or `uv run crawler-webhook`
- Tests (quick): `uv run pytest -m "not slow and not requires_services"`
- Full tests + coverage: `uv run pytest --cov=crawler_mcp --cov-report=term-missing`
- Lint/format: `uv run ruff check .` / `uv run ruff format .`
- Types: `uv run mypy crawler_mcp`
- Pre-commit: `uv run pre-commit install && uv run pre-commit run -a`

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indents, max line length 88, use double quotes.
- Enforce with Ruff (pycodestyle/pyflakes/isort/bugbear/naming/pyupgrade). Fix all warnings.
- Type hints required; mypy runs in strict mode inside `crawler_mcp/`.
- Naming: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.

## Testing Guidelines
- Framework: `pytest` + `pytest-asyncio` for async tests.
- Name tests `tests/test_*.py`. Async tests: `@pytest.mark.asyncio`.
- Useful selections: `uv run pytest -m unit` or `uv run pytest -m "not requires_services"`.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat:`, `fix:`, `refactor:`). Use imperative, concise subjects.
- PRs include: summary, rationale, linked issues, test coverage notes, screenshots/logs when relevant.
- Before review: run lint, format, types, and tests (see commands above).

## Security & Configuration Tips
- Never commit secrets. Copy `.env.example` to `.env` locally.
- Local services via `docker-compose up -d` (e.g., Qdrant, TEI). Default `SERVER_PORT=8010`.
- Prefer `uv run` for tools to use the locked environment.

## Agent-Specific Notes
- Install for Claude Desktop: `fastmcp install claude-desktop crawler_mcp/server.py`.
- Register new MCP tools under `crawler_mcp/tools/` and wire them in `register_*_tools` within the server.
