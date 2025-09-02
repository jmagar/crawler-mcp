from __future__ import annotations

from .crawl_cli import main

"""Thin wrapper to invoke the crawl CLI.

Kept for backward compatibility with existing commands that import/run
`crawler_mcp.crawlers.optimized.run`. The implementation now lives in
`crawler_mcp.crawlers.optimized.crawl_cli`.
"""

if __name__ == "__main__":
    main()
