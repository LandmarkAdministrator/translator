"""
Logging Module

Configures structured logging for the Church Audio Translator using loguru.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def setup_logger(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_rotation: str = "10 MB",
    log_retention: str = "7 days",
    console_output: bool = True,
) -> None:
    """
    Configure the logger with file and console handlers.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (relative to project root or absolute)
        log_rotation: When to rotate log files (e.g., "10 MB", "1 day")
        log_retention: How long to keep old log files
        console_output: Whether to output to console
    """
    # Remove default handler
    logger.remove()

    # Console output format
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # File output format (more detailed)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # Add console handler
    if console_output:
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            colorize=True,
        )

    # Add file handler if log_dir specified
    if log_dir:
        # Resolve log directory
        log_path = Path(log_dir)
        if not log_path.is_absolute():
            log_path = get_project_root() / log_path

        # Create log directory
        log_path.mkdir(parents=True, exist_ok=True)

        # Main log file
        logger.add(
            log_path / "translator_{time:YYYY-MM-DD}.log",
            format=file_format,
            level=log_level,
            rotation=log_rotation,
            retention=log_retention,
            compression="gz",
        )

        # Error-only log file
        logger.add(
            log_path / "errors_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="ERROR",
            rotation=log_rotation,
            retention=log_retention,
            compression="gz",
        )


def get_logger(name: str):
    """
    Get a logger instance with a specific name/context.

    Args:
        name: Name for the logger (typically module name)

    Returns:
        Logger instance bound with the name
    """
    return logger.bind(name=name)


# Module-level convenience functions
def debug(msg: str, **kwargs) -> None:
    """Log a debug message."""
    logger.debug(msg, **kwargs)


def info(msg: str, **kwargs) -> None:
    """Log an info message."""
    logger.info(msg, **kwargs)


def warning(msg: str, **kwargs) -> None:
    """Log a warning message."""
    logger.warning(msg, **kwargs)


def error(msg: str, **kwargs) -> None:
    """Log an error message."""
    logger.error(msg, **kwargs)


def critical(msg: str, **kwargs) -> None:
    """Log a critical message."""
    logger.critical(msg, **kwargs)


def exception(msg: str, **kwargs) -> None:
    """Log an exception with traceback."""
    logger.exception(msg, **kwargs)
