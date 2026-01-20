"""Enumeration types for domain models."""

from enum import StrEnum


class DataCategory(StrEnum):
    """Categories of data in the system."""
    
    # User data
    USERS = "users"
    
    # Content types
    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
    
    # Processed data
    CHUNKS = "chunks"
    EMBEDDINGS = "embeddings"
    
    # Datasets
    INSTRUCT_DATASET = "instruct_dataset"
    INSTRUCT_SAMPLES = "instruct_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"
    PREFERENCE_SAMPLES = "preference_dataset_samples"
    
    # Queries
    QUERIES = "queries"
    PROMPTS = "prompts"


class ContentType(StrEnum):
    """Types of content."""
    
    TEXT = "text"
    CODE = "code"
    MIXED = "mixed"


class Platform(StrEnum):
    """Social media and content platforms."""
    
    LINKEDIN = "linkedin"
    MEDIUM = "medium"
    GITHUB = "github"
    SUBSTACK = "substack"
    TWITTER = "twitter"
    OTHER = "other"


class ChunkStrategy(StrEnum):
    """Chunking strategies for text processing."""
    
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"


class EmbeddingModel(StrEnum):
    """Available embedding models."""
    
    MINI_LM = "sentence-transformers/all-MiniLM-L6-v2"
    MPNet = "sentence-transformers/all-mpnet-base-v2"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
