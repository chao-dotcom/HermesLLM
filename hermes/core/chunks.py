"""Chunk models for processed content."""

from abc import ABC

from pydantic import UUID4, Field

from .base import VectorDocument
from .enums import DataCategory, Platform


class Chunk(VectorDocument, ABC):
    """Base class for content chunks."""
    
    content: str = Field(description="Chunk text content")
    platform: Platform = Field(description="Source platform")
    document_id: UUID4 = Field(description="Parent document ID")
    author_id: UUID4 = Field(description="Author ID")
    author_full_name: str = Field(description="Author name")
    chunk_index: int = Field(0, description="Position in document")
    total_chunks: int = Field(1, description="Total chunks in document")
    
    @property
    def category(self) -> str:
        """Get data category from config."""
        if hasattr(self.Config, "category"):
            return self.Config.category
        return "chunks"


class ArticleChunk(Chunk):
    """Chunk from an article."""
    
    title: str | None = None
    link: str
    
    class Config:
        category = DataCategory.ARTICLES


class PostChunk(Chunk):
    """Chunk from a social media post."""
    
    image_url: str | None = Field(None, alias="image")
    link: str | None = None
    
    class Config:
        category = DataCategory.POSTS


class RepositoryChunk(Chunk):
    """Chunk from a code repository."""
    
    name: str
    link: str
    file_path: str | None = None
    language: str | None = None
    
    class Config:
        category = DataCategory.REPOSITORIES
