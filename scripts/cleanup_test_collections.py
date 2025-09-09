#!/usr/bin/env python3
"""
Script to clean up orphaned test collections from Qdrant.

This script will delete all collections that match test patterns while
preserving production collections like 'crawlerr_documents' and 'crawler_pages'.
"""

import asyncio
import logging
import sys

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
            logger.info(
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
            logger.warning(
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
                logger.warning(f"⚠️  SKIP: '{collection_name}' (already deleted)")
                return True
            else:
                logger.error(
                    f"❌ FAILED: '{collection_name}' - HTTP {response.status_code}: {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ ERROR: Failed to delete '{collection_name}': {e}")
            return False


async def delete_test_collections(test_collections: set[str]) -> None:
    """Delete all identified test collections."""
    if not test_collections:
        logger.info("No test collections to delete.")
        return

    logger.info(f"Starting deletion of {len(test_collections)} test collections...")

    success_count = 0
    failure_count = 0

    # Delete collections one by one to avoid overwhelming Qdrant
    for collection_name in sorted(test_collections):
        success = await delete_collection(collection_name)
        if success:
            success_count += 1
        else:
            failure_count += 1

        # Small delay to be gentle on Qdrant
        await asyncio.sleep(0.1)

    logger.info(f"Cleanup completed: {success_count} deleted, {failure_count} failed")


async def verify_cleanup() -> None:
    """Verify that cleanup was successful by listing remaining collections."""
    logger.info("Verifying cleanup...")

    remaining_collections = await get_all_collections()
    remaining_test_collections = identify_test_collections(remaining_collections)

    if remaining_test_collections:
        logger.warning(
            f"⚠️  {len(remaining_test_collections)} test collections still remain:"
        )
        for collection in sorted(remaining_test_collections):
            logger.warning(f"  - {collection}")
    else:
        logger.info("✅ Cleanup successful! No test collections remain.")

    logger.info(f"Final state: {len(remaining_collections)} total collections")
    for collection in sorted(remaining_collections):
        if collection in PROTECTED_COLLECTIONS:
            logger.info(f"  ✅ {collection} (protected)")
        else:
            logger.info(f"  🔍 {collection} (review needed)")


async def main():
    """Main cleanup function."""
    try:
        logger.info("Starting Qdrant test collection cleanup...")
        logger.info(f"Qdrant URL: {QDRANT_URL}")

        # Get all collections
        all_collections = await get_all_collections()

        if not all_collections:
            logger.info("No collections found. Nothing to clean up.")
            return

        # Identify test collections
        test_collections = identify_test_collections(all_collections)

        if not test_collections:
            logger.info("No test collections found. Nothing to clean up.")
            return

        # Show what will be deleted
        logger.info("Collections to be deleted:")
        for collection in sorted(test_collections):
            logger.info(f"  - {collection}")

        # Ask for confirmation
        print(
            f"\n⚠️  WARNING: This will delete {len(test_collections)} test collections!"
        )
        print("Protected collections that will be preserved:")
        for collection in sorted(PROTECTED_COLLECTIONS):
            print(f"  ✅ {collection}")

        # In CI/automated environments, skip confirmation
        if "--force" in sys.argv:
            logger.info("Force mode enabled, proceeding with deletion...")
        else:
            response = input("\nProceed with deletion? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                logger.info("Cleanup cancelled by user.")
                return

        # Delete test collections
        await delete_test_collections(test_collections)

        # Verify cleanup
        await verify_cleanup()

        logger.info("Qdrant test collection cleanup completed!")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Add help text
    if "--help" in sys.argv:
        print(__doc__)
        print("\nUsage:")
        print("  python cleanup_test_collections.py          # Interactive mode")
        print("  python cleanup_test_collections.py --force  # Skip confirmation")
        sys.exit(0)

    # Run cleanup
    asyncio.run(main())
