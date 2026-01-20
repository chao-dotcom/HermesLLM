"""Query models for retrieval."""

from pydantic import UUID4, Field

from atlas.core.base import VectorDocument
from atlas.core.enums import DataCategory


class Query(VectorDocument):
    """Query model for search."""
    
    content: str = Field(description="Query text content")
    author_id: UUID4 | None = None
    author_full_name: str | None = None
    
    class Config:
        category = DataCategory.QUERIES
    
    @classmethod
    def from_str(cls, query: str) -> "Query":
        """Create query from string."""
        return cls(content=query.strip("\n "), metadata={})
    
    def replace_content(self, new_content: str) -> "Query":
        """Create new query with replaced content."""
        return Query(
            id=self.id,
            content=new_content,
            author_id=self.author_id,
            author_full_name=self.author_full_name,
            metadata=self.metadata,
            vector=self.vector
        )


class EmbeddedQuery(Query):
    """Query with embedding vector."""
    
    embedding: list[float] = Field(description="Query embedding")
    
    class Config:
        category = DataCategory.QUERIES
