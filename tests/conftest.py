"""Test configuration and fixtures for HermesLLM.

This module provides pytest fixtures and utilities used across all tests.
"""

import os
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv

# Load test environment variables
env_file = os.getenv("ENV_FILE", ".env.testing")
if Path(env_file).exists():
    load_dotenv(env_file)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory) -> Path:
    """Create temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("test_artifacts")


@pytest.fixture
def sample_text() -> str:
    """Return sample text for testing."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on 
    developing systems that can learn from and make decisions based on data. 
    Large Language Models (LLMs) are a type of AI model trained on vast amounts 
    of text data to understand and generate human-like text.
    """


@pytest.fixture
def sample_documents() -> list[str]:
    """Return sample documents for testing."""
    return [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning algorithms can learn patterns from data without explicit programming.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Transformers are a type of neural network architecture for sequence processing.",
    ]


@pytest.fixture
def sample_metadata() -> dict:
    """Return sample metadata for testing."""
    return {
        "author": "test_author",
        "source": "test_source",
        "timestamp": "2024-01-01T00:00:00",
        "tags": ["test", "example"],
    }


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    # Set test environment variables
    test_env = {
        "MONGODB_DATABASE_NAME": "test_db",
        "QDRANT_COLLECTION_NAME": "test_collection",
        "OPENAI_API_KEY": "test-key",
        "COMET_API_KEY": "test-comet-key",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    # Reimport settings to pick up test values
    from hermes.config import settings
    return settings


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Mock OpenAI client for testing."""
    class MockOpenAI:
        def __init__(self, *args, **kwargs):
            pass
        
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    class MockResponse:
                        class Choice:
                            class Message:
                                content = "This is a mock response from OpenAI."
                            message = Message()
                        choices = [Choice()]
                    return MockResponse()
    
    monkeypatch.setattr("openai.OpenAI", MockOpenAI)
    return MockOpenAI


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Mock embedding model for testing."""
    import numpy as np
    
    class MockEmbeddings:
        def __init__(self, *args, **kwargs):
            self.embedding_dim = 384
        
        def encode(self, texts, **kwargs):
            """Return random embeddings."""
            if isinstance(texts, str):
                texts = [texts]
            return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)
    
    return MockEmbeddings()


@pytest.fixture
def mock_qdrant_client(monkeypatch):
    """Mock Qdrant client for testing."""
    class MockQdrantClient:
        def __init__(self, *args, **kwargs):
            self.collections = {}
        
        def get_collections(self):
            class Collections:
                collections = []
            return Collections()
        
        def create_collection(self, collection_name, vectors_config):
            self.collections[collection_name] = []
        
        def upsert(self, collection_name, points):
            if collection_name not in self.collections:
                self.collections[collection_name] = []
            self.collections[collection_name].extend(points)
        
        def search(self, collection_name, query_vector, limit=5, **kwargs):
            # Return mock search results
            return [
                type('obj', (object,), {
                    'id': i,
                    'score': 0.9 - i * 0.1,
                    'payload': {'text': f'Document {i}'}
                })()
                for i in range(min(limit, 3))
            ]
    
    return MockQdrantClient()


@pytest.fixture
def mock_mongodb_client(monkeypatch):
    """Mock MongoDB client for testing."""
    class MockCollection:
        def __init__(self):
            self.documents = []
        
        def insert_one(self, document):
            self.documents.append(document)
            class InsertResult:
                inserted_id = "test_id"
            return InsertResult()
        
        def insert_many(self, documents):
            self.documents.extend(documents)
            class InsertResult:
                inserted_ids = [f"test_id_{i}" for i in range(len(documents))]
            return InsertResult()
        
        def find(self, *args, **kwargs):
            return iter(self.documents)
        
        def find_one(self, *args, **kwargs):
            return self.documents[0] if self.documents else None
        
        def count_documents(self, *args, **kwargs):
            return len(self.documents)
        
        def delete_many(self, *args, **kwargs):
            count = len(self.documents)
            self.documents = []
            class DeleteResult:
                deleted_count = count
            return DeleteResult()
    
    class MockDatabase:
        def __getitem__(self, collection_name):
            return MockCollection()
    
    class MockMongoClient:
        def __getitem__(self, database_name):
            return MockDatabase()
        
        def close(self):
            pass
    
    return MockMongoClient()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Add cleanup logic here if needed


# Skip markers for conditional test execution
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "skipif_no_api: skip if API credentials not available"
    )
    config.addinivalue_line(
        "markers", "skipif_no_db: skip if database not available"
    )
    config.addinivalue_line(
        "markers", "skipif_no_aws: skip if AWS credentials not available"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers."""
    # Skip tests requiring API if credentials not available
    if not os.getenv("OPENAI_API_KEY"):
        skip_api = pytest.mark.skip(reason="API credentials not available")
        for item in items:
            if "requires_api" in item.keywords:
                item.add_marker(skip_api)
    
    # Skip tests requiring database if not available
    if not os.getenv("MONGODB_URI"):
        skip_db = pytest.mark.skip(reason="Database not available")
        for item in items:
            if "requires_db" in item.keywords:
                item.add_marker(skip_db)
    
    # Skip tests requiring AWS if credentials not available
    if not os.getenv("AWS_ACCESS_KEY"):
        skip_aws = pytest.mark.skip(reason="AWS credentials not available")
        for item in items:
            if "requires_aws" in item.keywords:
                item.add_marker(skip_aws)
