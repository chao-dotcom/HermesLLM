"""Pydantic schemas for API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Collection schemas
class CollectionRequest(BaseModel):
    """Request to collect data from URLs."""
    
    user_id: str = Field(..., description="User ID")
    full_name: str = Field(..., description="User's full name")
    urls: List[str] = Field(..., description="List of URLs to collect")


class CollectionResponse(BaseModel):
    """Response from collection operation."""
    
    status: str = Field(..., description="Operation status")
    total: int = Field(..., description="Total URLs processed")
    success: int = Field(..., description="Successfully collected")
    failed: int = Field(..., description="Failed to collect")


# Processing schemas
class ProcessingRequest(BaseModel):
    """Request to process documents."""
    
    author_id: Optional[str] = Field(None, description="Filter by author ID")
    limit: Optional[int] = Field(None, description="Maximum documents to process")
    chunk_size: int = Field(500, description="Chunk size for splitting")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )


class ProcessingResponse(BaseModel):
    """Response from processing operation."""
    
    status: str
    documents_processed: int
    chunks_created: int
    embeddings_generated: int
    vectors_stored: int


# RAG schemas
class RAGQueryRequest(BaseModel):
    """Request for RAG query."""
    
    query: str = Field(..., description="Query text")
    use_query_expansion: bool = Field(False, description="Whether to expand query")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    top_k: int = Field(5, description="Number of chunks to retrieve")


class RAGQueryResponse(BaseModel):
    """Response from RAG query."""
    
    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Health schemas
class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    services: Dict[str, str]
