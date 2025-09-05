# GitHub Webhook Server Module

This module provides GitHub organization webhook processing for automatic AI prompt extraction from PR comments and reviews, integrated into the Crawler MCP server.

## Overview

The webhook server automatically processes GitHub organization webhook events to extract:
- 🤖 **CodeRabbit AI prompts** from review comments
- 💼 **Copilot suggestions** from PR reviews
- 📝 **Committable suggestions** with code blocks
- 🧠 **Any comment containing code blocks** from relevant bots

## Architecture

The webhook server runs alongside the main MCP server in the same Docker container using supervisor:

```
GitHub Org Webhook → SWAG Reverse Proxy → Docker Container
                                              ├── MCP Server (port 8010)
                                              └── Webhook Server (port 38080)
                                                    ↓
                                              Extract AI Prompts
                                                    ↓
                                              /output/pr/
```

## Configuration

All configuration is done via environment variables in `.env`:

### Required Variables
```bash
# GitHub webhook secret (set in GitHub organization settings)
GITHUB_WEBHOOK_SECRET=your-webhook-secret-here

# GitHub API token for accessing PR data
GITHUB_TOKEN=ghp_your-github-token-here
```

### Optional Variables
```bash
# Server configuration
WEBHOOK_PORT=38080

# Repository filtering (* for all, or comma-separated list)
REPOS_TO_TRACK=*

# Processing settings
WEBHOOK_SCRIPT_PATH=./scripts/extract_coderabbit_prompts.py
WEBHOOK_OUTPUT_DIR=./output
WEBHOOK_MAX_CONCURRENT_PROCESSES=5

# Event type filtering
PROCESS_REVIEWS=true
PROCESS_REVIEW_COMMENTS=true
PROCESS_ISSUE_COMMENTS=true

# Bot pattern matching
BOT_PATTERNS=coderabbitai[bot],copilot-pull-request-reviewer[bot],Copilot
```

## Files Structure

```
crawler_mcp/webhook/
├── __init__.py           # Module exports
├── server.py             # FastAPI webhook server
└── README.md            # This documentation
```

## Usage

The webhook server is automatically started when running the combined Docker container:

```bash
# Start all services (MCP + Webhook + Qdrant + TEI)
docker-compose up -d

# Check webhook server status
curl http://localhost:38080/health

# View webhook processing stats
curl http://localhost:38080/stats
```

## SWAG Reverse Proxy Configuration

Create `/config/nginx/proxy-confs/webhook.subdomain.conf`:

```nginx
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name webhook.*;
    include /config/nginx/ssl.conf;
    client_max_body_size 10M;

    location / {
        include /config/nginx/proxy.conf;
        include /config/nginx/resolver.conf;

        set $upstream_app crawler-mcp-server;  # Docker container name
        set $upstream_port 38080;
        set $upstream_proto http;

        proxy_pass $upstream_proto://$upstream_app:$upstream_port;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }
}
```

## GitHub Organization Setup

1. Go to: `https://github.com/organizations/YOUR_ORG/settings/hooks`
2. Click **Add webhook**
3. Configure:
   - **Payload URL**: `https://webhook.yourdomain.com/webhook`
   - **Content type**: `application/json`
   - **Secret**: Your `GITHUB_WEBHOOK_SECRET` value
4. Select events:
   - ✅ Pull request reviews
   - ✅ Pull request review comments
   - ✅ Issue comments
5. Click **Add webhook**

## API Endpoints

The webhook server exposes these endpoints:

### `POST /webhook`
Main webhook endpoint that GitHub calls. Handles signature verification and event processing.

### `GET /recent`
Get recent PRs from all accessible repositories:
```bash
# Get PRs from last 7 days (default)
curl http://localhost:38080/recent

# Get PRs from last 14 days
curl "http://localhost:38080/recent?days=14"

# Using the CLI script to list only
python scripts/batch_extraction.py --list-only --days 7
```

Response:
```json
{
  "recent_prs": [
    {
      "repo": "owner/repo",
      "pr_number": 123,
      "title": "Add new feature",
      "state": "open",
      "updated_at": "2025-01-25T10:30:00Z",
      "url": "https://github.com/owner/repo/pull/123"
    }
  ],
  "total_prs": 1,
  "days_back": 7,
  "message": "Found 1 PRs updated in the last 7 days"
}
```

### `POST /batch`
Trigger batch extraction for multiple PRs:
```bash
# Auto-discover and process recent PRs
curl -X POST http://localhost:38080/batch \
  -H "Content-Type: application/json" \
  -d '{"auto_discover": true, "days": 7}'

# Process specific PRs
curl -X POST http://localhost:38080/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prs": [
      {"owner": "jmagar", "repo": "crawler-mcp", "pr_number": 10},
      {"owner": "jmagar", "repo": "docker-mcp", "pr_number": 5}
    ]
  }'

# Using the CLI script for auto-discovery
python scripts/batch_extraction.py --auto-discover --days 7

# With custom webhook URL
python scripts/batch_extraction.py --auto-discover --url https://githook.tootie.tv
```

Expected JSON payload (auto-discovery):
```json
{
  "auto_discover": true,
  "days": 7
}
```

Expected JSON payload (specific PRs):
```json
{
  "prs": [
    {"owner": "user", "repo": "repo1", "pr_number": 123},
    {"owner": "user", "repo": "repo2", "pr_number": 456}
  ]
}
```

Response:
```json
{
  "status": "batch_queued",
  "total_prs": 2,
  "successfully_queued": 2,
  "failed": 0,
  "tasks": [
    {"repo": "user/repo1", "pr_number": 123, "status": "queued"},
    {"repo": "user/repo2", "pr_number": 456, "status": "queued"}
  ],
  "message": "Queued extraction for 2 PRs"
}
```

### `POST /manual`
Manually trigger extraction for a specific PR:
```bash
# Using curl
curl -X POST http://localhost:38080/manual \
  -H "Content-Type: application/json" \
  -d '{"owner": "jmagar", "repo": "crawler-mcp", "pr_number": 10}'

# Using the CLI script
python scripts/manual_extraction.py jmagar crawler-mcp 10

# With custom webhook URL
python scripts/manual_extraction.py jmagar crawler-mcp 10 --url https://githook.tootie.tv
```

Expected JSON payload:
```json
{
  "owner": "github-username",
  "repo": "repository-name",
  "pr_number": 123
}
```

Response:
```json
{
  "status": "queued",
  "repo": "owner/repo",
  "pr_number": 123,
  "message": "Extraction queued for owner/repo#123"
}
```

### `GET /health`
Health check endpoint:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-25T17:00:00.000000",
  "stats": {
    "total_webhooks": 42,
    "processed_events": 38,
    "failed_events": 0,
    "active_processes": 2
  },
  "queue_size": 0,
  "active_processes": 2
}
```

### `GET /stats`
Processing statistics:
```json
{
  "stats": {
    "total_webhooks": 42,
    "processed_events": 38,
    "failed_events": 0,
    "active_processes": 2
  },
  "queue_size": 0,
  "active_processes": ["owner/repo#123"],
  "config": {
    "repos_tracked": "*",
    "max_concurrent": 5,
    "bot_patterns": ["coderabbitai[bot]", "copilot-pull-request-reviewer[bot]"]
  }
}
```

### `GET /`
Service information endpoint with available endpoints list.

## Event Processing Flow

1. **Webhook Received**: GitHub sends POST to `/webhook`
2. **Signature Verification**: HMAC SHA-256 signature validation
3. **Event Filtering**: Check if event type and repository should be processed
4. **Content Analysis**: Verify comment contains relevant AI content
5. **Queue Processing**: Add to background processing queue
6. **Script Execution**: Run extraction script asynchronously
7. **Output Storage**: Save results to `/output/pr/` directory

## Output Files

PR analysis data is saved in structured format in `output/pr/`:

```
output/pr/
├── myrepo-123/
│   ├── items.ndjson
│   ├── suggestions.json
│   └── resolved.json
├── myrepo-124/
│   ├── items.ndjson
│   ├── suggestions.json
│   └── resolved.json
└── ...
```

Each PR directory contains:
- `items.ndjson`: Structured PR data (comments, reviews, metadata)
- `suggestions.json`: Extracted code suggestions and fixes
- `resolved.json`: Resolution status and applied changes

## Error Handling

- **Invalid signature**: Returns 401 Unauthorized
- **Invalid JSON**: Returns 400 Bad Request
- **Processing errors**: Logged but webhook returns 200 OK
- **Script failures**: Logged with stderr output
- **Queue backlog**: Tasks wait if concurrent limit reached

## Monitoring

Monitor webhook processing via:

```bash
# Container logs
docker-compose logs -f crawler-mcp

# Webhook server specific logs
docker exec crawler-mcp-server tail -f /app/logs/webhook-server.log

# Health check
curl https://webhook.yourdomain.com/health

# Processing stats
curl https://webhook.yourdomain.com/stats
```

## Security Features

- ✅ **Webhook signature verification** prevents unauthorized requests
- ✅ **Non-root container execution** for security
- ✅ **Input validation** prevents injection attacks
- ✅ **Rate limiting** via concurrent process limits
- ✅ **Environment-based secrets** management
- ✅ **HTTPS-only** via SWAG reverse proxy

## Integration with MCP Server

The webhook outputs can be integrated with MCP tools (future enhancement):

```python
# Future: MCP tools to query webhook outputs
@mcp.tool
async def list_ai_prompts() -> List[str]:
    """List extracted AI prompt files."""
    return list(Path("output/pr").glob("*/items.ndjson"))

@mcp.tool
async def get_ai_prompts(pr_number: int) -> str:
    """Get AI prompts for specific PR."""
    file_path = Path(f"output/pr/{{repo}}-{pr_number}/items.ndjson")
    return file_path.read_text() if file_path.exists() else "Not found"
```

## Troubleshooting

**No webhook events received:**
- Check GitHub webhook delivery logs in organization settings
- Verify SWAG proxy configuration and SSL certificates
- Confirm webhook URL accessibility: `curl https://webhook.yourdomain.com/health`

**Invalid signature errors:**
- Verify `GITHUB_WEBHOOK_SECRET` matches GitHub organization webhook settings
- Check for whitespace in environment variables

**Script execution failures:**
- Verify `WEBHOOK_SCRIPT_PATH` points to correct extraction script
- Check `GITHUB_TOKEN` has appropriate repository permissions
- Review script logs in `/app/logs/webhook-server.log`

**High resource usage:**
- Reduce `WEBHOOK_MAX_CONCURRENT_PROCESSES` value
- Filter repositories using `REPOS_TO_TRACK` if not all repos need processing
- Disable unnecessary event types (reviews, comments, etc.)
