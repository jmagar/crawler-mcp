"""
Custom response models for optimized crawler.

This module provides wrapper classes that extend Crawl4AI's AsyncCrawlResponse
to include additional attributes expected by tests and downstream consumers.
"""

import logging
from typing import Any

from crawl4ai.models import AsyncCrawlResponse

logger = logging.getLogger(__name__)


class OptimizedCrawlResponse(AsyncCrawlResponse):
    """
    Extended AsyncCrawlResponse with additional attributes for optimized crawler.

    This class adds the attributes that tests and consumers expect:
    - success: Boolean indicating crawl success
    - extracted_content: The main text content extracted from pages
    - metadata: Dictionary with crawl statistics and information
    """

    def __init__(self, **data):
        """
        Initialize OptimizedCrawlResponse.

        Args:
            **data: All AsyncCrawlResponse fields plus additional custom fields
        """
        # Extract custom fields and store them as regular attributes
        success = data.pop("success", True)
        extracted_content = data.pop("extracted_content", "")
        metadata = data.pop("metadata", {})

        # Initialize parent class with remaining data
        super().__init__(**data)

        # Set custom attributes after initialization to avoid Pydantic conflicts
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "extracted_content", extracted_content)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_async_crawl_response(
        cls,
        response: AsyncCrawlResponse,
        success: bool = True,
        extracted_content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "OptimizedCrawlResponse":
        """
        Create OptimizedCrawlResponse from a standard AsyncCrawlResponse.

        Args:
            response: Original AsyncCrawlResponse
            success: Whether the crawl was successful
            extracted_content: Extracted text content
            metadata: Additional metadata about the crawl

        Returns:
            OptimizedCrawlResponse with all data from original response plus custom fields
        """
        # Extract all data from original response
        response_data = {}
        for field in response.model_fields:
            if hasattr(response, field):
                response_data[field] = getattr(response, field)

        # Add custom fields
        response_data.update(
            {
                "success": success,
                "extracted_content": extracted_content,
                "metadata": metadata or {},
            }
        )

        return cls(**response_data)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert response to dictionary for serialization.

        Returns:
            Dictionary representation of the response
        """
        result = {}

        # Include all parent fields
        for field in self.model_fields:
            if hasattr(self, field):
                value = getattr(self, field)
                result[field] = value

        # Include custom fields
        result.update(
            {
                "success": self.success,
                "extracted_content": self.extracted_content,
                "metadata": self.metadata,
            }
        )

        return result
