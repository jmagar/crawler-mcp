# Multi-stage build for crawler_mcp with integrated webhook server
FROM python:3.11-slim as builder

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install uv and sync dependencies
RUN pip install --no-cache-dir uv && \
    uv sync --frozen && \
    ls -la .venv/bin/ && \
    .venv/bin/python --version && \
    .venv/bin/pip list | grep fastmcp

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r crawler && useradd -r -g crawler crawler

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY crawler_mcp/ ./crawler_mcp/
COPY scripts/ ./scripts/
COPY .env.example ./

# Create directories for outputs and logs
RUN mkdir -p webhook_outputs logs data && \
    chown -R crawler:crawler /app && \
    chmod +x /app/.venv/bin/*

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set PATH to include virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER crawler

# Expose ports
EXPOSE 8010 38080

# Health check for both services
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8010/health && curl -f http://localhost:38080/health || exit 1

# Run supervisor to manage both services
USER root
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
