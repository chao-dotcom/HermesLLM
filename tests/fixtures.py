"""Test fixtures and utilities for HermesLLM tests.

This module provides reusable fixtures and helper functions for testing.
"""

import pytest
from pathlib import Path
from typing import Generator
import tempfile
import shutil


@pytest.fixture
def temp_test_dir() -> Generator[Path, None, None]:
    """Create temporary test directory that gets cleaned up."""
    temp_dir = Path(tempfile.mkdtemp(prefix="hermes_test_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_raw_documents():
    """Return sample raw documents for testing."""
    return [
        {
            "id": "doc-1",
            "content": "First test document about machine learning.",
            "author_id": "author-1",
            "platform": "medium",
        },
        {
            "id": "doc-2",
            "content": "Second test document about deep learning.",
            "author_id": "author-1",
            "platform": "github",
        },
        {
            "id": "doc-3",
            "content": "Third test document about natural language processing.",
            "author_id": "author-2",
            "platform": "linkedin",
        },
    ]


@pytest.fixture
def sample_chunks():
    """Return sample chunks for testing."""
    return [
        {
            "id": "chunk-1",
            "content": "Machine learning is a subset of AI.",
            "document_id": "doc-1",
            "chunk_index": 0,
        },
        {
            "id": "chunk-2",
            "content": "Deep learning uses neural networks.",
            "document_id": "doc-2",
            "chunk_index": 0,
        },
        {
            "id": "chunk-3",
            "content": "NLP processes human language.",
            "document_id": "doc-3",
            "chunk_index": 0,
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Return sample embeddings for testing."""
    import numpy as np
    
    # 384-dimensional embeddings
    return np.random.rand(3, 384).astype(np.float32)


@pytest.fixture
def mock_llm_response():
    """Return mock LLM response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a mock LLM response for testing purposes."
                }
            }
        ]
    }


@pytest.fixture
def sample_training_data():
    """Return sample training data."""
    return [
        {
            "instruction": "What is machine learning?",
            "input": "",
            "output": "Machine learning is a subset of AI that enables systems to learn from data.",
        },
        {
            "instruction": "Explain deep learning.",
            "input": "",
            "output": "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
        },
    ]


@pytest.fixture
def sample_queries():
    """Return sample queries for testing."""
    return [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain natural language processing.",
        "What are transformers in AI?",
    ]


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
    
    def encode(self, texts, **kwargs):
        """Generate random embeddings."""
        import numpy as np
        
        if isinstance(texts, str):
            texts = [texts]
        
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, response="Mock LLM response"):
        self.response = response
    
    def generate(self, prompt, **kwargs):
        """Generate mock response."""
        return self.response
    
    def __call__(self, prompt, **kwargs):
        """Allow callable usage."""
        return self.generate(prompt, **kwargs)


@pytest.fixture
def mock_embedding_model():
    """Return mock embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
def mock_llm():
    """Return mock LLM."""
    return MockLLM()


def create_test_file(directory: Path, filename: str, content: str) -> Path:
    """Helper to create test files.
    
    Args:
        directory: Directory to create file in
        filename: Name of file
        content: Content to write
        
    Returns:
        Path to created file
    """
    file_path = directory / filename
    file_path.write_text(content)
    return file_path


def assert_valid_embedding(embedding, expected_dim=384):
    """Helper to validate embeddings.
    
    Args:
        embedding: Embedding to validate
        expected_dim: Expected dimension
    """
    import numpy as np
    
    assert isinstance(embedding, (list, np.ndarray))
    
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    
    assert len(embedding.shape) in [1, 2]
    
    if len(embedding.shape) == 1:
        assert embedding.shape[0] == expected_dim
    else:
        assert embedding.shape[1] == expected_dim


def assert_valid_chunk(chunk_dict):
    """Helper to validate chunk structure.
    
    Args:
        chunk_dict: Chunk dictionary to validate
    """
    required_fields = ["id", "content", "document_id", "chunk_index"]
    
    for field in required_fields:
        assert field in chunk_dict, f"Missing required field: {field}"
    
    assert isinstance(chunk_dict["content"], str)
    assert isinstance(chunk_dict["chunk_index"], int)
    assert chunk_dict["chunk_index"] >= 0
