"""Log management for the optimized crawler.

Provides rotating log files with size-based rotation.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


class LogManager:
    """Manages crawler logging with automatic rotation."""

    def __init__(self, log_dir: str = "./output/logs"):
        """Initialize logging configuration.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log files
        self.crawl_log_path = self.log_dir / "crawl.log"
        self.error_log_path = self.log_dir / "errors.log"

        # Rotation settings
        self.max_bytes = 10 * 1024 * 1024  # 10MB
        self.backup_count = 3  # Keep 3 rotated versions

        # Initialize loggers
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup base logging configuration."""
        # Clear existing handlers to avoid duplicates
        logging.getLogger().handlers.clear()

        # Set base logging level
        logging.getLogger().setLevel(logging.INFO)

    def setup_crawl_logger(self, name: str = "crawl") -> logging.Logger:
        """Setup main crawl logger with rotation.

        Args:
            name: Logger name

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        # Remove existing handlers
        logger.handlers.clear()

        # Create rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            filename=self.crawl_log_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )

        # Set format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent duplicate messages

        return logger

    def setup_error_logger(self, name: str = "errors") -> logging.Logger:
        """Setup error-specific logger with rotation.

        Args:
            name: Logger name

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        # Remove existing handlers
        logger.handlers.clear()

        # Create rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            filename=self.error_log_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )

        # Set format with more detail for errors
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)  # Only log errors and above
        logger.propagate = False  # Prevent duplicate messages

        return logger

    def setup_console_logger(self, name: str = "console") -> logging.Logger:
        """Setup console logger for terminal output.

        Args:
            name: Logger name

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        # Remove existing handlers
        logger.handlers.clear()

        # Create console handler
        handler = logging.StreamHandler()

        # Simple format for console
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """Get logger by name with appropriate handlers.

        Args:
            name: Logger type ('crawl', 'errors', 'console')

        Returns:
            Configured logger instance
        """
        if name == "crawl":
            return self.setup_crawl_logger()
        elif name == "errors":
            return self.setup_error_logger()
        elif name == "console":
            return self.setup_console_logger()
        else:
            # Default to crawl logger
            return self.setup_crawl_logger(name)

    def get_log_file_sizes(self) -> dict[str, int]:
        """Get current log file sizes in bytes.

        Returns:
            Dictionary with file sizes
        """
        sizes = {}

        for name, path in [
            ("crawl", self.crawl_log_path),
            ("errors", self.error_log_path),
        ]:
            if path.exists():
                sizes[name] = path.stat().st_size
            else:
                sizes[name] = 0

        return sizes

    def get_log_info(self) -> dict[str, any]:
        """Get information about log files and rotation.

        Returns:
            Dictionary with log information
        """
        info = {
            "log_dir": str(self.log_dir),
            "max_size_mb": self.max_bytes / (1024 * 1024),
            "backup_count": self.backup_count,
            "files": {},
        }

        # Check each log file and its rotations
        for log_name, log_path in [
            ("crawl", self.crawl_log_path),
            ("errors", self.error_log_path),
        ]:
            file_info = {"current_size": 0, "rotations": []}

            # Current file
            if log_path.exists():
                file_info["current_size"] = log_path.stat().st_size

            # Check for rotated files
            for i in range(1, self.backup_count + 1):
                rotated_path = Path(f"{log_path}.{i}")
                if rotated_path.exists():
                    file_info["rotations"].append(
                        {
                            "number": i,
                            "size": rotated_path.stat().st_size,
                            "modified": rotated_path.stat().st_mtime,
                        }
                    )

            info["files"][log_name] = file_info

        return info

    def cleanup_old_logs(self, keep_days: int = 7) -> None:
        """Remove log files older than specified days.

        Args:
            keep_days: Number of days to keep logs
        """
        import time

        cutoff_time = time.time() - (keep_days * 24 * 3600)

        # Check all log files in the directory
        for file_path in self.log_dir.glob("*.log*"):
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
            except OSError:
                continue

    def force_rotation(self, log_type: str = "both") -> None:
        """Force log rotation for specified log type.

        Args:
            log_type: "crawl", "errors", or "both"
        """
        if log_type in ("crawl", "both"):
            logger = self.get_logger("crawl")
            for handler in logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()

        if log_type in ("errors", "both"):
            logger = self.get_logger("errors")
            for handler in logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()
