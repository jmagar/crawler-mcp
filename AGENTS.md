# Repository Guidelines

## Project Structure & Module Organization
- `crawler_mcp/`: core package (server, config, tools, middleware, core services, webhook).
- `tests/`: pytest suite with asyncio markers; integration and slow tests are marked.
- `scripts/`: utility scripts for batch/manual extraction and helpers.
- `docs/`, `examples/`, `data/`, `logs/`: ancillary assets and outputs.
- Key entry points: `crawler_mcp/crawlers/optimized/server.py` (MCP server), `crawler_mcp/webhook/server.py` (webhook).

## Build, Test, and Development Commands
- Install (editable): `uv sync` or `pip install -e .`
- Run server (dev): `fastmcp dev crawler_mcp/crawlers/optimized/server.py`
- Run server (direct): `uv run python -m crawler_mcp.server` or `uv run crawler-mcp`
- Webhook (direct): `uv run python -m crawler_mcp.webhook.server` or `uv run crawler-webhook`
- Tests (quick): `uv run pytest -m "not slow and not requires_services"`
- Full tests + coverage: `uv run pytest --cov=crawler_mcp --cov-report=term-missing`
- Lint: `uv run ruff check .`  Format: `uv run ruff format .`
- Types: `uv run mypy crawler_mcp`
- Pre-commit: `uv run pre-commit install && uv run pre-commit run -a`

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indents, max line length 88, double quotes.
- Use ruff (pycodestyle/pyflakes/isort/bugbear/naming/pyupgrade); fix all warnings.
- Type hints required (mypy strict in package; allowlisted externals are ignored).
- Names: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Keep public APIs in `tools/` minimal and documented via docstrings.

## Testing Guidelines
- Framework: pytest with `pytest-asyncio`; markers: `unit`, `integration`, `slow`, `requires_services`.
- Place tests under `tests/` and name `test_*.py`; async tests use `@pytest.mark.asyncio`.
- CI-friendly selection examples:
  - Unit only: `uv run pytest -m unit`
  - Exclude externals: `uv run pytest -m "not requires_services"`

## Commit & Pull Request Guidelines
- Commit style: Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.); imperative, concise subject.
- Example: `refactor: optimize RAG system architecture and performance`.
- PRs must include: summary, rationale, linked issues, test coverage notes, and screenshots/logs when relevant.
- Required checks before requesting review: lint, format, types, tests (see commands above).

## Security & Configuration Tips
- Do not commit secrets. Copy `.env.example` to `.env` and adjust locally.
- Local services via `docker-compose up -d` (Qdrant, TEI). Server defaults to `SERVER_PORT=8010`.
- Prefer `uv run` for all tool invocations to use the locked environment.

## Agent-Specific Notes
- Install for Claude: `fastmcp install claude-desktop crawler_mcp/crawlers/optimized/server.py`.
- Exposed MCP tools are registered in `crawler_mcp/crawlers/optimized/server.py`; add new tools under `crawler_mcp/crawlers/optimized/tools/` and register via `register_*_tools`.
