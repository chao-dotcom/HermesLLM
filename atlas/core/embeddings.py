"""Embedded chunk models with vector representations."""

from abc import ABC

from pydantic import UUID4, Field

from .base import VectorDocument
from .enums import DataCategory, Platform


class EmbeddedChunk(VectorDocument, ABC):
    """Base class for chunks with embeddings."""
    
    content: str = Field(description="Chunk text content")
    embedding: list[float] | None = Field(None, alias="vector")
    platform: Platform = Field(description="Source platform")
    document_id: UUID4 = Field(description="Parent document ID")
    author_id: UUID4 = Field(description="Author ID")
    author_full_name: str = Field(description="Author name")
    chunk_index: int = Field(0, description="Position in document")
    
    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"], max_chunks: int | None = None) -> str:
        """
        Convert chunks to context string for RAG.
        
        Args:
            chunks: List of embedded chunks
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Formatted context string
        """
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_context = f"""
Chunk {i}:
Type: {chunk.__class__.__name__.replace('Embedded', '').replace('Chunk', '')}
Platform: {chunk.platform}
Author: {chunk.author_full_name}
Content: {chunk.content}
"""
            context_parts.append(chunk_context.strip())
        
        return "\n\n".join(context_parts)
    
    @property
    def collection_name(self) -> str:
        """Get Qdrant collection name."""
        if hasattr(self.Config, "name"):
            return self.Config.name
        return f"embedded_{self.__class__.__name__.lower()}"


class EmbeddedArticleChunk(EmbeddedChunk):
    """Article chunk with embedding."""
    
    title: str | None = None
    link: str
    
    class Config:
        name = "embedded_articles"
        category = DataCategory.ARTICLES
        use_vector_index = True


class EmbeddedPostChunk(EmbeddedChunk):
    """Post chunk with embedding."""
    
    image_url: str | None = None
    link: str | None = None
    
    class Config:
        name = "embedded_posts"
        category = DataCategory.POSTS
        use_vector_index = True


class EmbeddedRepositoryChunk(EmbeddedChunk):
    """Repository chunk with embedding."""
    
    name: str
    link: str
    file_path: str | None = None
    language: str | None = None
    
    class Config:
        name = "embedded_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = True


class Query(VectorDocument):
    """Query with embedding for retrieval."""
    
    text: str = Field(description="Query text")
    embedding: list[float] | None = Field(None, alias="vector")
    expanded_queries: list[str] = Field(default_factory=list)
    metadata_filters: dict = Field(default_factory=dict)
    
    @classmethod
    def from_text(cls, text: str) -> "Query":
        """Create query from text."""
        return cls(text=text, metadata={})
