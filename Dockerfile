# syntax=docker/dockerfile:1.5
# Multi-stage build for crawler_mcp with integrated webhook server
FROM python:3.11-slim as builder

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install uv and sync dependencies with cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir uv && \
    uv sync --frozen

FROM python:3.11-slim as runtime

# Create non-root user with home directory (matching host UID/GID)
RUN groupadd -g 1000 crawler && \
    useradd -u 1000 -g 1000 -m -d /home/crawler crawler

WORKDIR /app

# Install system dependencies including Playwright browser dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    supervisor \
    # Playwright browser dependencies
    libglib2.0-0 \
    libnspr4 \
    libnss3 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libcairo2 \
    libpango-1.0-0 \
    libasound2 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment with ownership set during copy (avoids slow chown)
COPY --from=builder --chown=crawler:crawler /app/.venv /app/.venv

# Pre-create directories with correct ownership
RUN mkdir -p /app/logs /app/data /app/data/.crawl4ai \
    /home/crawler/.cache/ms-playwright && \
    chown -R crawler:crawler /app/logs /app/data \
    /home/crawler/.cache/ms-playwright

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV HOME="/home/crawler"
ENV PLAYWRIGHT_BROWSERS_PATH="/home/crawler/.cache/ms-playwright"

# Install Playwright browsers as crawler user
USER crawler
RUN playwright install chromium

# Switch back to root for remaining operations
USER root

# Copy application code with ownership set during copy
COPY --chown=crawler:crawler crawler_mcp/ ./crawler_mcp/
COPY --chown=crawler:crawler scripts/ ./scripts/
COPY --chown=crawler:crawler .env.example ./
COPY --chown=crawler:crawler entrypoint.sh ./

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set executable permissions only (avoid recursive chown)
RUN chmod +x /app/entrypoint.sh /app/.venv/bin/*

# Expose ports
EXPOSE 8010 38080

# Health check for both services
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8010/health && curl -f http://localhost:38080/health || exit 1

# Run entrypoint script to fix permissions then start supervisor
CMD ["/app/entrypoint.sh"]
