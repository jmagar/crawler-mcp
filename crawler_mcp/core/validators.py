"""
Unified validation framework for consistent data validation across the application.

This module provides standardized validators to replace the 37+ validation functions
with similar patterns found throughout the codebase.
"""

import os
import re
import urllib.parse
from collections.abc import Collection
from pathlib import Path
from re import Pattern
from typing import Any

from .exceptions import ValidationError
from .logging import get_logger

logger = get_logger(__name__)


class ValidationResult:
    """Result of a validation operation."""

    def __init__(
        self, is_valid: bool, error_message: str | None = None, value: Any = None
    ):
        self.is_valid = is_valid
        self.error_message = error_message
        self.value = value

    def __bool__(self) -> bool:
        return self.is_valid

    @classmethod
    def success(cls, value: Any = None) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(True, None, value)

    @classmethod
    def failure(cls, error_message: str, value: Any = None) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(False, error_message, value)


class BaseValidator:
    """Base class for all validators."""

    def __init__(self, error_message: str | None = None, required: bool = True):
        self.error_message = error_message
        self.required = required

    def validate(self, value: Any, field_name: str = "value") -> ValidationResult:
        """
        Validate a value.

        Args:
            value: Value to validate
            field_name: Name of the field being validated

        Returns:
            ValidationResult
        """
        # Handle None/empty values based on required flag
        if value is None or (isinstance(value, str) and not value.strip()):
            if self.required:
                return ValidationResult.failure(
                    self.error_message or f"{field_name} is required"
                )
            else:
                return ValidationResult.success(value)

        return self._validate_value(value, field_name)

    def _validate_value(self, value: Any, field_name: str) -> ValidationResult:
        """Override in subclasses to implement specific validation logic."""
        return ValidationResult.success(value)

    def __call__(self, value: Any, field_name: str = "value") -> ValidationResult:
        """Allow validator to be called as a function."""
        return self.validate(value, field_name)


class StringValidator(BaseValidator):
    """Validator for string values."""

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | Pattern[str] | None = None,
        choices: Collection[str] | None = None,
        strip: bool = True,
        lowercase: bool = False,
        error_message: str | None = None,
        required: bool = True,
    ):
        super().__init__(error_message, required)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.choices = set(choices) if choices else None
        self.strip = strip
        self.lowercase = lowercase

    def _validate_value(self, value: Any, field_name: str) -> ValidationResult:
        # Convert to string
        if not isinstance(value, str):
            value = str(value)

        # Apply transformations
        if self.strip:
            value = value.strip()
        if self.lowercase:
            value = value.lower()

        # Check empty after transformations
        if self.required and not value:
            return ValidationResult.failure(
                self.error_message or f"{field_name} cannot be empty"
            )

        # Length validation
        if self.min_length is not None and len(value) < self.min_length:
            return ValidationResult.failure(
                self.error_message
                or f"{field_name} must be at least {self.min_length} characters"
            )

        if self.max_length is not None and len(value) > self.max_length:
            return ValidationResult.failure(
                self.error_message
                or f"{field_name} must be at most {self.max_length} characters"
            )

        # Pattern validation
        if self.pattern and not self.pattern.match(value):
            return ValidationResult.failure(
                self.error_message or f"{field_name} does not match required pattern"
            )

        # Choices validation
        if self.choices and value not in self.choices:
            return ValidationResult.failure(
                self.error_message
                or f"{field_name} must be one of: {', '.join(self.choices)}"
            )

        return ValidationResult.success(value)


class NumericValidator(BaseValidator):
    """Validator for numeric values."""

    def __init__(
        self,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
        allow_negative: bool = True,
        allow_zero: bool = True,
        integer_only: bool = False,
        error_message: str | None = None,
        required: bool = True,
    ):
        super().__init__(error_message, required)
        self.min_value = min_value
        self.max_value = max_value
        self.allow_negative = allow_negative
        self.allow_zero = allow_zero
        self.integer_only = integer_only

    def _validate_value(self, value: Any, field_name: str) -> ValidationResult:
        # Convert to numeric
        try:
            if self.integer_only:
                if isinstance(value, float) and not value.is_integer():
                    return ValidationResult.failure(f"{field_name} must be an integer")
                value = int(value)
            else:
                value = float(value)
        except (ValueError, TypeError):
            return ValidationResult.failure(f"{field_name} must be a number")

        # Sign validation
        if not self.allow_negative and value < 0:
            return ValidationResult.failure(f"{field_name} cannot be negative")

        if not self.allow_zero and value == 0:
            return ValidationResult.failure(f"{field_name} cannot be zero")

        # Range validation
        if self.min_value is not None and value < self.min_value:
            return ValidationResult.failure(
                f"{field_name} must be at least {self.min_value}"
            )

        if self.max_value is not None and value > self.max_value:
            return ValidationResult.failure(
                f"{field_name} must be at most {self.max_value}"
            )

        return ValidationResult.success(value)


class PathValidator(BaseValidator):
    """Validator for file system paths."""

    def __init__(
        self,
        must_exist: bool = True,
        must_be_file: bool | None = None,
        must_be_directory: bool | None = None,
        must_be_readable: bool = False,
        must_be_writable: bool = False,
        allowed_extensions: Collection[str] | None = None,
        error_message: str | None = None,
        required: bool = True,
    ):
        super().__init__(error_message, required)
        self.must_exist = must_exist
        self.must_be_file = must_be_file
        self.must_be_directory = must_be_directory
        self.must_be_readable = must_be_readable
        self.must_be_writable = must_be_writable
        self.allowed_extensions = (
            set(allowed_extensions) if allowed_extensions else None
        )

    def _validate_value(self, value: Any, field_name: str) -> ValidationResult:
        # Convert to Path
        try:
            path = Path(str(value))
        except Exception:
            return ValidationResult.failure(f"{field_name} is not a valid path")

        # Existence check
        if self.must_exist and not path.exists():
            return ValidationResult.failure(f"{field_name} does not exist: {path}")

        if path.exists():
            # File/directory type checks
            if self.must_be_file and not path.is_file():
                return ValidationResult.failure(f"{field_name} must be a file: {path}")

            if self.must_be_directory and not path.is_dir():
                return ValidationResult.failure(
                    f"{field_name} must be a directory: {path}"
                )

            # Permission checks using os.access() for accurate permission testing
            if self.must_be_readable and not os.access(path, os.R_OK):
                return ValidationResult.failure(
                    f"{field_name} must be readable: {path}"
                )

            if self.must_be_writable:
                # Check parent directory writability for non-existing files
                check_path = path if path.exists() else path.parent
                if not check_path.exists():
                    return ValidationResult.failure(
                        f"{field_name} parent directory does not exist: {path}"
                    )
                if not os.access(check_path, os.W_OK):
                    return ValidationResult.failure(
                        f"{field_name} must be writable: {check_path}"
                    )

        # Extension check
        if (
            self.allowed_extensions
            and path.suffix.lower() not in self.allowed_extensions
        ):
            return ValidationResult.failure(
                f"{field_name} must have one of these extensions: {', '.join(self.allowed_extensions)}"
            )

        return ValidationResult.success(path)


class UrlValidator(BaseValidator):
    """Validator for URL values."""

    def __init__(
        self,
        schemes: Collection[str] | None = None,
        require_netloc: bool = True,
        check_accessibility: bool = False,
        error_message: str | None = None,
        required: bool = True,
    ):
        super().__init__(error_message, required)
        self.schemes = set(schemes) if schemes else {"http", "https"}
        self.require_netloc = require_netloc
        self.check_accessibility = check_accessibility

    def _validate_value(self, value: Any, field_name: str) -> ValidationResult:
        url_str = str(value).strip()

        # Parse URL
        try:
            parsed = urllib.parse.urlparse(url_str)
        except Exception:
            return ValidationResult.failure(f"{field_name} is not a valid URL")

        # Scheme validation
        if parsed.scheme not in self.schemes:
            return ValidationResult.failure(
                f"{field_name} must use one of these schemes: {', '.join(self.schemes)}"
            )

        # Netloc validation
        if self.require_netloc and not parsed.netloc:
            return ValidationResult.failure(f"{field_name} must include a domain")

        # TODO: Add accessibility check if needed (would require async)
        # This could be implemented as an async validator subclass

        return ValidationResult.success(url_str)


class CollectionValidator(BaseValidator):
    """Validator for collections (lists, sets, etc.)."""

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        item_validator: BaseValidator | None = None,
        unique_items: bool = False,
        error_message: str | None = None,
        required: bool = True,
    ):
        super().__init__(error_message, required)
        self.min_length = min_length
        self.max_length = max_length
        self.item_validator = item_validator
        self.unique_items = unique_items

    def _validate_value(self, value: Any, field_name: str) -> ValidationResult:
        # Ensure it's a collection
        if not hasattr(value, "__len__") or isinstance(value, str | bytes):
            return ValidationResult.failure(f"{field_name} must be a collection")

        # Length validation
        if self.min_length is not None and len(value) < self.min_length:
            return ValidationResult.failure(
                f"{field_name} must contain at least {self.min_length} items"
            )

        if self.max_length is not None and len(value) > self.max_length:
            return ValidationResult.failure(
                f"{field_name} must contain at most {self.max_length} items"
            )

        # Uniqueness check
        if self.unique_items:
            unique_items = (
                set(value)
                if all(
                    isinstance(item, str | int | float | bool | type(None))
                    for item in value
                )
                else []
            )
            if len(unique_items) != len(value):
                return ValidationResult.failure(
                    f"{field_name} must contain unique items"
                )

        # Validate individual items
        if self.item_validator:
            validated_items = []
            for i, item in enumerate(value):
                result = self.item_validator.validate(item, f"{field_name}[{i}]")
                if not result.is_valid:
                    return result
                validated_items.append(result.value)
            return ValidationResult.success(validated_items)

        return ValidationResult.success(value)


class CompositeValidator(BaseValidator):
    """Validator that combines multiple validators with AND/OR logic."""

    def __init__(
        self,
        validators: Collection[BaseValidator],
        logic: str = "and",  # "and" or "or"
        error_message: str | None = None,
        required: bool = True,
    ):
        super().__init__(error_message, required)
        self.validators = validators
        self.logic = logic.lower()

    def _validate_value(self, value: Any, field_name: str) -> ValidationResult:
        results = [
            validator.validate(value, field_name) for validator in self.validators
        ]

        if self.logic == "and":
            # All validators must pass
            for result in results:
                if not result.is_valid:
                    return result
            return ValidationResult.success(value)

        elif self.logic == "or":
            # At least one validator must pass
            for result in results:
                if result.is_valid:
                    return result
            # All failed, return the first error
            return (
                results[0]
                if results
                else ValidationResult.failure("No validators provided")
            )

        else:
            return ValidationResult.failure("Invalid logic mode for CompositeValidator")


class AsyncValidatorMixin:
    """Mixin for creating async validators. Must be used with BaseValidator."""

    def validate(self, value: Any, field_name: str = "value") -> ValidationResult:
        """Sync validation method that should be implemented by the base class."""
        raise NotImplementedError("validate method must be implemented")

    async def async_validate(
        self, value: Any, field_name: str = "value"
    ) -> ValidationResult:
        """Async validation method to be overridden in subclasses."""
        # Default implementation falls back to sync validation
        return self.validate(value, field_name)


# Convenience factory functions for common validation patterns
def required_string(
    min_length: int = 1, max_length: int | None = None
) -> StringValidator:
    """Create a validator for required non-empty strings."""
    return StringValidator(min_length=min_length, max_length=max_length)


def optional_string(max_length: int | None = None) -> StringValidator:
    """Create a validator for optional strings."""
    return StringValidator(max_length=max_length, required=False)


def positive_integer(max_value: int | None = None) -> NumericValidator:
    """Create a validator for positive integers."""
    return NumericValidator(
        min_value=1,
        max_value=max_value,
        allow_negative=False,
        allow_zero=False,
        integer_only=True,
    )


def non_negative_integer(max_value: int | None = None) -> NumericValidator:
    """Create a validator for non-negative integers."""
    return NumericValidator(
        min_value=0, max_value=max_value, allow_negative=False, integer_only=True
    )


def integer_range(min_value: int, max_value: int) -> NumericValidator:
    """Create a validator for integers within a specific range."""
    return NumericValidator(min_value=min_value, max_value=max_value, integer_only=True)


def existing_file(extensions: Collection[str] | None = None) -> PathValidator:
    """Create a validator for existing files."""
    return PathValidator(
        must_exist=True, must_be_file=True, allowed_extensions=extensions
    )


def existing_directory() -> PathValidator:
    """Create a validator for existing directories."""
    return PathValidator(must_exist=True, must_be_directory=True)


def http_url() -> UrlValidator:
    """Create a validator for HTTP/HTTPS URLs."""
    return UrlValidator(schemes={"http", "https"})


def non_empty_list(item_validator: BaseValidator | None = None) -> CollectionValidator:
    """Create a validator for non-empty lists."""
    return CollectionValidator(min_length=1, item_validator=item_validator)


# Validation utilities
def validate_and_raise(
    validator: BaseValidator, value: Any, field_name: str = "value"
) -> Any:
    """
    Validate a value and raise ValidationError if invalid.

    Args:
        validator: Validator to use
        value: Value to validate
        field_name: Name of the field being validated

    Returns:
        Validated value (possibly transformed)

    Raises:
        ValidationError: If validation fails
    """
    result = validator.validate(value, field_name)
    if not result.is_valid:
        raise ValidationError(result.error_message or "Validation failed")
    return result.value


def validate_dict(
    validators: dict[str, BaseValidator], data: dict[str, Any]
) -> dict[str, Any]:
    """
    Validate a dictionary of values using a mapping of field validators.

    Args:
        validators: Mapping of field names to validators
        data: Data to validate

    Returns:
        Dictionary of validated values

    Raises:
        ValidationError: If any validation fails
    """
    validated_data = {}

    for field_name, validator in validators.items():
        value = data.get(field_name)
        result = validator.validate(value, field_name)

        if not result.is_valid:
            raise ValidationError(f"Field '{field_name}': {result.error_message}")

        # Only include the field if it has a value or is required
        if result.value is not None or validator.required:
            validated_data[field_name] = result.value

    return validated_data


# Pre-configured validators for common use cases in the application
class CrawlerValidators:
    """Collection of pre-configured validators for the crawler application."""

    # URL validation
    crawl_url = UrlValidator(schemes={"http", "https"})

    # Numeric limits commonly used in the crawler
    max_pages = NumericValidator(min_value=1, max_value=2000, integer_only=True)
    max_depth = NumericValidator(min_value=1, max_value=5, integer_only=True)

    # Worker/concurrency limits
    embedding_workers = integer_range(1, 16)
    browser_pool_size = integer_range(1, 20)

    # File system paths
    directory_path = existing_directory()
    file_path = existing_file()

    # Text content validation
    non_empty_text = required_string(min_length=1)
    query_text = required_string(min_length=1, max_length=1000)

    # Collection validation
    file_patterns = CollectionValidator(
        min_length=0, item_validator=StringValidator(min_length=1), required=False
    )
