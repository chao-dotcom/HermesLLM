"""Testing utilities for HermesLLM.

This module provides helper functions and utilities for testing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_test_data(filename: str) -> Any:
    """Load test data from JSON file.
    
    Args:
        filename: Name of test data file
        
    Returns:
        Loaded test data
    """
    test_data_dir = Path(__file__).parent / "data"
    file_path = test_data_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    
    with file_path.open() as f:
        return json.load(f)


def save_test_data(filename: str, data: Any) -> None:
    """Save test data to JSON file.
    
    Args:
        filename: Name of test data file
        data: Data to save
    """
    test_data_dir = Path(__file__).parent / "data"
    test_data_dir.mkdir(exist_ok=True)
    
    file_path = test_data_dir / filename
    
    with file_path.open('w') as f:
        json.dump(data, f, indent=2)


def compare_dicts(dict1: Dict, dict2: Dict, ignore_keys: List[str] = None) -> bool:
    """Compare two dictionaries, optionally ignoring certain keys.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        ignore_keys: Keys to ignore in comparison
        
    Returns:
        True if dictionaries are equal (excluding ignored keys)
    """
    ignore_keys = ignore_keys or []
    
    # Filter out ignored keys
    filtered_dict1 = {k: v for k, v in dict1.items() if k not in ignore_keys}
    filtered_dict2 = {k: v for k, v in dict2.items() if k not in ignore_keys}
    
    return filtered_dict1 == filtered_dict2


def assert_embeddings_similar(emb1, emb2, threshold=0.1):
    """Assert that two embeddings are similar.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
        threshold: Similarity threshold (lower = more similar)
    """
    import numpy as np
    
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    
    # Cosine similarity
    dot_product = np.dot(emb1, emb2)
    norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    similarity = dot_product / norm_product
    
    assert similarity > (1 - threshold), f"Embeddings not similar enough: {similarity}"


def count_tokens(text: str) -> int:
    """Count tokens in text (simple approximation).
    
    Args:
        text: Text to count tokens in
        
    Returns:
        Approximate token count
    """
    # Simple approximation: ~4 chars per token
    return len(text) // 4


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for display in tests.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def create_mock_response(content: str, metadata: Dict = None) -> Dict:
    """Create mock API response.
    
    Args:
        content: Response content
        metadata: Optional metadata
        
    Returns:
        Mock response dictionary
    """
    return {
        "content": content,
        "metadata": metadata or {},
        "timestamp": "2024-01-01T00:00:00Z",
    }


class TestLogger:
    """Simple logger for capturing test output."""
    
    def __init__(self):
        self.messages = []
    
    def info(self, message: str):
        """Log info message."""
        self.messages.append(("INFO", message))
    
    def error(self, message: str):
        """Log error message."""
        self.messages.append(("ERROR", message))
    
    def warning(self, message: str):
        """Log warning message."""
        self.messages.append(("WARNING", message))
    
    def get_messages(self, level: str = None) -> List[str]:
        """Get logged messages.
        
        Args:
            level: Optional level to filter by
            
        Returns:
            List of messages
        """
        if level:
            return [msg for lvl, msg in self.messages if lvl == level]
        return [msg for _, msg in self.messages]
    
    def clear(self):
        """Clear all messages."""
        self.messages = []


def skip_if_no_api_key(key_name: str):
    """Decorator to skip test if API key not available.
    
    Args:
        key_name: Name of environment variable for API key
    """
    import os
    import pytest
    
    def decorator(func):
        if not os.getenv(key_name):
            return pytest.mark.skip(
                reason=f"API key {key_name} not available"
            )(func)
        return func
    
    return decorator


def requires_package(package_name: str):
    """Decorator to skip test if package not installed.
    
    Args:
        package_name: Name of required package
    """
    import pytest
    import importlib.util
    
    def decorator(func):
        if importlib.util.find_spec(package_name) is None:
            return pytest.mark.skip(
                reason=f"Package {package_name} not installed"
            )(func)
        return func
    
    return decorator
