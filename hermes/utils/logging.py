"""Logging configuration."""

import sys

from loguru import logger


def setup_logging(level: str = "INFO", format_string: str | None = None) -> None:
    """
    Configure loguru logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string
    """
    logger.remove()  # Remove default handler
    
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True
    )
    
    logger.info(f"Logging initialized at {level} level")
