#!/usr/bin/env python3
"""
Maintenance script for periodic cleanup of test collections in Qdrant.

This script can be run periodically (e.g., via cron job) to clean up any
orphaned test collections that may have accumulated due to test failures
or incomplete cleanup.
"""

import asyncio
import logging
import sys
from datetime import datetime

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Qdrant connection settings
QDRANT_URL = "http://localhost:7000"

# Production collections to preserve (never delete these)
PROTECTED_COLLECTIONS = {
    "crawlerr_documents",
    "crawler_pages",
    "documents_test",  # Keep this one as it might be intentional
}

# Test patterns that identify collections to delete
TEST_PATTERNS = [
    "test_crawler_",
    "test_github_pr_",
    "test_aggressive_",
    "test_conservative_",
    "test_distance_cosine_",
    "batch_test_",
    "scale_test_",
    "reconnect_test_",
    "multipage_test_",
    "recovery_test_",
    "concurrent_test_",
    "rag_test_",
    "github_test_",
    "e2e_test_",
    "filtered_rag_",
    "docs_test_",
    "perf_test_",
    "test_collection_",  # From the fixture
]


async def get_all_collections() -> list[str]:
    """Get list of all collections from Qdrant."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{QDRANT_URL}/collections")
            response.raise_for_status()

            data = response.json()
            collections = [col["name"] for col in data["result"]["collections"]]
            logger.info(f"Found {len(collections)} total collections")
            return collections

        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            raise


def identify_test_collections(all_collections: list[str]) -> set[str]:
    """Identify which collections are test collections that should be deleted."""
    test_collections = set()

    for collection_name in all_collections:
        # Skip protected collections
        if collection_name in PROTECTED_COLLECTIONS:
            logger.debug(
                f"PROTECTED: Skipping '{collection_name}' (production collection)"
            )
            continue

        # Check if collection matches any test pattern
        is_test_collection = False
        for pattern in TEST_PATTERNS:
            if collection_name.startswith(pattern):
                test_collections.add(collection_name)
                is_test_collection = True
                logger.debug(
                    f"MATCHED: '{collection_name}' matches pattern '{pattern}'"
                )
                break

        if not is_test_collection:
            logger.info(
                f"UNKNOWN: Collection '{collection_name}' doesn't match any known pattern"
            )

    logger.info(f"Identified {len(test_collections)} test collections for deletion")
    return test_collections


async def delete_collection(collection_name: str) -> bool:
    """Delete a single collection from Qdrant."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(
                f"{QDRANT_URL}/collections/{collection_name}"
            )

            if response.status_code == 200:
                logger.info(f"✅ DELETED: '{collection_name}'")
                return True
            elif response.status_code == 404:
                logger.info(f"⚠️  SKIP: '{collection_name}' (already deleted)")
                return True
            else:
                logger.error(
                    f"❌ FAILED: '{collection_name}' - HTTP {response.status_code}: {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ ERROR: Failed to delete '{collection_name}': {e}")
            return False


async def maintenance_cleanup() -> None:
    """Perform maintenance cleanup of test collections."""
    try:
        logger.info("Starting Qdrant maintenance cleanup...")
        logger.info(f"Qdrant URL: {QDRANT_URL}")

        # Get all collections
        all_collections = await get_all_collections()

        if not all_collections:
            logger.info("No collections found. Nothing to clean up.")
            return

        # Identify test collections
        test_collections = identify_test_collections(all_collections)

        if not test_collections:
            logger.info("No test collections found. Qdrant is clean!")
            return

        # Show what will be deleted
        logger.info(f"Found {len(test_collections)} test collections to clean up:")
        for collection in sorted(test_collections):
            logger.info(f"  - {collection}")

        # Delete test collections
        logger.info("Starting deletion...")
        success_count = 0
        failure_count = 0

        for collection_name in sorted(test_collections):
            success = await delete_collection(collection_name)
            if success:
                success_count += 1
            else:
                failure_count += 1

            # Small delay to be gentle on Qdrant
            await asyncio.sleep(0.1)

        logger.info(
            f"Maintenance cleanup completed: {success_count} deleted, {failure_count} failed"
        )

        # Final verification
        remaining_collections = await get_all_collections()
        remaining_test_collections = identify_test_collections(remaining_collections)

        if remaining_test_collections:
            logger.warning(
                f"⚠️  {len(remaining_test_collections)} test collections still remain"
            )
        else:
            logger.info(
                "✅ Maintenance cleanup successful! No test collections remain."
            )

        logger.info(
            f"Final state: {len(remaining_collections)} total collections (should be production only)"
        )

    except Exception as e:
        logger.error(f"Maintenance cleanup failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    try:
        # Add help text
        if "--help" in sys.argv or "-h" in sys.argv:
            print(__doc__)
            print(
                "\nThis script performs maintenance cleanup of test collections in Qdrant."
            )
            print(
                "It's safe to run periodically and will only delete test collections."
            )
            print("\nProtected collections (never deleted):")
            for collection in sorted(PROTECTED_COLLECTIONS):
                print(f"  ✅ {collection}")
            sys.exit(0)

        # Run maintenance cleanup
        logger.info(f"Qdrant Maintenance Cleanup started at {datetime.now()}")
        asyncio.run(maintenance_cleanup())
        logger.info(f"Qdrant Maintenance Cleanup completed at {datetime.now()}")

    except KeyboardInterrupt:
        logger.info("Maintenance cleanup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Maintenance cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
