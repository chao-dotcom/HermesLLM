"""Helper utilities."""

import time
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

from loguru import logger

T = TypeVar("T")


def batch(items: list[T], batch_size: int) -> Iterator[list[T]]:
    """
    Yield successive batches from items.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each attempt
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_filename(name: str) -> str:
    """
    Convert string to safe filename.
    
    Args:
        name: Original name
        
    Returns:
        Safe filename
    """
    import re
    
    # Replace invalid characters with underscore
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Remove multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Strip leading/trailing underscores
    return safe.strip("_")
