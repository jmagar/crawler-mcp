from __future__ import annotations

from .cli.crawl import main

"""Thin wrapper to invoke the crawl CLI.

Kept for backward compatibility with existing commands that import/run
`crawler_mcp.crawlers.optimized.run`. The implementation now lives in
`crawler_mcp.crawlers.optimized.cli.crawl`.
"""

if __name__ == "__main__":
    main()
