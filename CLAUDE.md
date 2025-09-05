# Crawler MCP - FastMCP RAG-Enabled Web Crawling Server Project Memory

This file provides guidance to Claude Code (claude.ai/code) for developing this Crawler MCP FastMCP server.

## FastMCP Framework Knowledge
- **Primary Transport**: Streamable-HTTP is recommended for most servers.
- **Package Manager**: `uv` is the standard for this project. Use `pyproject.toml` to manage dependencies.
- **Core Dependencies**: Key libraries include `fastmcp`, `crawl4ai`, `qdrant-client`, `pydantic`, `httpx`, `transformers`, and `sentence-transformers`. Optional: `tiktoken` (fallback tokenizer).

## Development Guidelines

### Recommended Project Structure
Follow this structure for a clean and maintainable server.
```
crawler_mcp/
├── .claude/                  # Claude Code specific files (optional)
│   └── templates/            # Reusable templates for the agent
├── .env.example              # Example environment variables (for local config)
├── .env                      # Local environment variables
├── .gitignore
├── CLAUDE.md                 # This file
├── IMPLEMENTATION_PLAN.md    # Detailed implementation specifications
├── docker-compose.yml        # Qdrant and HF TEI services
├── crawler_mcp/              # Main application package
│   ├── __init__.py
│   ├── config.py           # Pydantic settings management
│   ├── server.py           # Main server entry point
│   ├── prompts/            # Reusable crawling and analysis prompt templates
│   │   └── __init__.py
│   ├── middleware/         # FastMCP middleware (logging, error handling, progress tracking)
│   │   └── __init__.py
│   ├── models/             # Pydantic data models
│   │   └── __init__.py
│   ├── core/               # Core services (embeddings, vectors, rag, sources)
│   │   └── __init__.py
│   ├── crawlers/           # Crawler implementations (web, repository, directory)
│   │   └── __init__.py
│   └── tools/              # MCP tool implementations (scrape, crawl, rag_query, etc.)
│       └── __init__.py
├── pyproject.toml            # Project dependencies and metadata
├── README.md                 # Project overview and setup instructions
└── tests/
    └── test_server.py
```

### Dependency Management with `uv`
- Use `uv add <package>` to add a new dependency. This updates `pyproject.toml`.
- After modifying dependencies, run `uv sync` to update your environment.
- Always commit the `uv.lock` file to source control to ensure reproducible builds.

### Configuration Management
- Use Pydantic's `BaseSettings` in a `config.py` file to manage settings.
- Load configuration from environment variables and `.env` files for flexibility across different environments.

### Development Workflow with `fastmcp` CLI
Use the built-in `fastmcp` command-line interface for an efficient development cycle.

#### Interactive Debugging (`fastmcp dev`)
For interactive testing and debugging, use the `dev` command. This runs your server with the MCP Inspector UI, which allows you to call tools and inspect responses.

- **Start the dev server**: `fastmcp dev crawler_mcp/crawlers/optimized/server.py --with crawl4ai --with qdrant-client --with torch`

#### Running the Server (`fastmcp run`)
To run the server directly (e.g., for integration testing or production), use the `run` command.

- **Run with HTTP transport**: `fastmcp run crawler_mcp/crawlers/optimized/server.py --transport http`
- **Run with dependencies**: `fastmcp run crawler_mcp/crawlers/optimized/server.py --with crawl4ai --with qdrant-client`

## Installation & Deployment
Once your server is ready, use the `fastmcp install` command to make it available to MCP clients.

- **For supported clients**: Use `fastmcp install <client_name> crawler_mcp/crawlers/optimized/server.py` (e.g., `claude-code`, `claude-desktop`). This handles dependency management with `uv` automatically.
- **For other clients**: Generate a standard configuration file using `fastmcp install mcp-json crawler_mcp/crawlers/optimized/server.py > mcp_config.json`. This file can be used with any MCP-compatible client.

## FastMCP Specific Patterns

- **Component Decorators**: Use `@mcp.tool`, `@mcp.resource("uri://path")`, and `@mcp.prompt` to define your server's capabilities.

- **Separation of Concerns**: Keep core business logic in the `core/` directory and crawler implementations in `crawlers/`. The MCP tool function should be a thin wrapper that handles context logging and calls the business logic. This makes the code more testable and reusable.

- **Middleware**: Use middleware to add cross-cutting concerns. For development, new servers should include `ErrorHandlingMiddleware`, `LoggingMiddleware`, and `TimingMiddleware` by default. Add them using `mcp.add_middleware()`.

- **Logging**: The framework uses a dual-logging strategy. 1) For operational feedback intended for the MCP client, use the context logger: `await ctx.info("Processing started")`. 2) For internal, server-side diagnostics (e.g., writing to a file), use Python's standard `logging` module.

- **Error Handling**: For predictable, operational errors that the client should act on (e.g., invalid input), `raise` a specific exception like `ToolError`. For all other unexpected errors, allow them to be caught by the `ErrorHandlingMiddleware` to prevent leaking implementation details.

## Advanced Development Patterns

- **Progress Reporting**: For any tool that performs a multi-step or iterative process, provide feedback to the client using `await ctx.report_progress(progress=i, total=total)`. This is crucial for a good user experience in long-running tasks.

- **User Elicitation**: For tools that require interactive user input, use `await ctx.elicit(message="...", response_type=...)`. This allows a tool to pause and ask for clarification or missing parameters. Always check the `result.action` to handle `accept`, `decline`, and `cancel` responses.

- **LLM Sampling**: To leverage the client's LLM for text generation or analysis within a tool, use `await ctx.sample(...)`. This is useful for offloading complex reasoning or content generation tasks to the LLM.

- **Accessing HTTP Requests**: When running with a web transport, use the `get_http_request()` dependency function to get the full Starlette `Request` object for headers, client IP, etc. This is useful for implementing logic based on request metadata.

- **FastAPI/Starlette Integration**: To embed the MCP server in a web app, mount it using `app.mount('/mcp', mcp.http_app())`. It is critical to manage the application lifecycle correctly by creating a `combined_lifespan` async context manager to handle startup and shutdown events for both applications.

- **Tool Transformation**: Use `Tool.from_tool()` to adapt existing tools. This is useful for simplifying complex tools (e.g., from an OpenAPI spec) or creating specialized variations of a tool without duplicating code.

## Leveraging External MCP Servers
This project can leverage a powerful arsenal of external MCP servers to aid in development, research, and automation. Make sure to utilize these tools when appropriate.

- **`github`**: For interacting with GitHub repositories (issues, PRs, code).
- **`github-chat`**: For chat-based interactions related to GitHub.
- **`context7`**: For structured documentation and knowledge base research.
- **`searxng`**: For comprehensive web searches.
- **`playwright`**: For advanced web automation and scraping.
- **`mcp-deepwiki`**: For deep research within wiki-like knowledge bases.
- **`deep-directory-tree`**: For analyzing and navigating complex directory structures.
- **`sequential-thinking`**: For structuring complex problems and multi-step reasoning.
- **`youtube-vision`**: For extracting information and insights from YouTube videos.
- **`gemini-coding`**: For general coding assistance and problem-solving.

## Testing Standards
- **Rapid Debugging**: Use `fastmcp dev` for interactive testing and debugging of your server.
- **Automated Testing**: Use `pytest` for comprehensive unit and integration tests. The most efficient way to test is **in-memory**, by passing the server instance directly to a `fastmcp.Client` within your test functions. This avoids running a separate server process.
- **Running Tests**: Execute your test suite with `uv run pytest`.

## Security Considerations
- Always be mindful of security. For production servers, ensure you have an authentication and authorization strategy in place.
- Sanitize all inputs, especially those used in shell commands or database queries, to prevent injection attacks.
