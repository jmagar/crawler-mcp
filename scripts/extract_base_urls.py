#!/usr/bin/env python3
"""
Extract unique base URLs from the Qdrant vector database.

This script connects to the Qdrant database, extracts all source URLs,
converts them to base URLs (domain + first path segment), and creates
a markdown file with the unique base URLs and their document counts.
"""

import asyncio
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from urllib.parse import urlparse

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from crawler_mcp.core.vectors.statistics import StatisticsCollector  # noqa: E402
from crawler_mcp.settings import get_settings  # noqa: E402


def extract_base_url(full_url: str) -> str:
    """
    Extract the base URL from a full URL.

    Examples:
    - https://ui.shadcn.com/docs/components/button → ui.shadcn.com/docs
    - https://github.com/user/repo/blob/main/file.py → github.com/user/repo
    - https://docs.python.org/3/library/asyncio.html → docs.python.org/3
    """
    try:
        parsed = urlparse(full_url)

        # Start with the domain
        base = parsed.netloc

        # Add the first path segment if it exists
        if parsed.path and parsed.path != "/":
            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                # For common cases, include more meaningful path segments
                if parsed.netloc == "github.com" and len(path_parts) >= 2:
                    # For GitHub, include user/repo
                    base += f"/{path_parts[0]}/{path_parts[1]}"
                elif parsed.netloc.startswith("docs.") and len(path_parts) >= 1:
                    # For docs sites, include version or main path
                    base += f"/{path_parts[0]}"
                else:
                    # General case: include first path segment
                    base += f"/{path_parts[0]}"

        return base
    except Exception:
        # Return the original URL if parsing fails
        return full_url


async def extract_all_base_urls() -> dict[str, int]:
    """
    Extract all unique base URLs from the Qdrant database.

    Returns:
        Dictionary mapping base URL to document count
    """
    settings = get_settings()
    stats_collector = StatisticsCollector()

    base_url_counts: dict[str, int] = defaultdict(int)

    try:
        # Scroll through all points to collect source URLs
        offset = None
        limit = 1000
        processed_count = 0

        print("Extracting URLs from Qdrant database...")

        while True:
            client = await stats_collector._get_client()
            result = await client.scroll(
                collection_name=settings.qdrant_collection,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # We don't need embeddings
            )

            points = result[0]
            next_offset = result[1]

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", None)
                if payload is None:
                    continue

                source_url = payload.get("source_url", "")
                if source_url:
                    base_url = extract_base_url(source_url)
                    base_url_counts[base_url] += 1
                    processed_count += 1

            print(f"Processed {processed_count} documents...", end="\r")

            if next_offset is None:
                break
            offset = next_offset

        print(f"\nProcessed {processed_count} total documents.")
        print(f"Found {len(base_url_counts)} unique base URLs.")

    except Exception as e:
        print(f"Error extracting URLs: {e}")
        return {}

    finally:
        # Clean up the client
        await stats_collector.close()

    return dict(base_url_counts)


def create_markdown_report(base_url_counts: dict[str, int], output_path: str) -> None:
    """
    Create a markdown report with the base URLs.

    Args:
        base_url_counts: Dictionary mapping base URL to document count
        output_path: Path to save the markdown file
    """
    # Sort URLs by domain, then by path
    sorted_urls = sorted(
        base_url_counts.items(), key=lambda x: (x[0].split("/")[0], x[0])
    )

    total_urls = len(sorted_urls)
    total_documents = sum(base_url_counts.values())

    # Create the markdown content
    content = f"""# Vector Database Base URLs

**Generated:** {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}

## Summary Statistics

- **Total unique base URLs:** {total_urls:,}
- **Total documents:** {total_documents:,}
- **Average documents per base URL:** {total_documents / max(1, total_urls):.1f}

## Base URLs by Domain

"""

    # Group by domain for better organization
    current_domain = None
    for base_url, count in sorted_urls:
        domain = base_url.split("/")[0]

        if domain != current_domain:
            content += f"\n### {domain}\n\n"
            current_domain = domain

        content += f"- **{base_url}** ({count:,} documents)\n"

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nMarkdown report saved to: {output_path}")


async def main() -> None:
    """Main execution function."""
    print("🕷️ Crawler MCP - Base URL Extractor")
    print("=" * 50)

    # Extract base URLs
    base_url_counts = await extract_all_base_urls()

    if not base_url_counts:
        print("No URLs found or extraction failed.")
        return

    # Create output path
    output_path = os.path.join(project_root, "vector_db_base_urls.md")

    # Create the markdown report
    create_markdown_report(base_url_counts, output_path)

    print("\n✅ Base URL extraction completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
