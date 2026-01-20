"""Document models for content storage."""

from abc import ABC
from typing import Any

from pydantic import UUID4, Field

from .base import MongoDocument
from .enums import DataCategory, Platform


class UserDocument(MongoDocument):
    """User profile document."""
    
    first_name: str
    last_name: str
    email: str | None = None
    bio: str | None = None
    
    class Settings:
        name = DataCategory.USERS
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"


class Document(MongoDocument, ABC):
    """Base class for content documents."""
    
    content: dict[str, Any] = Field(description="Structured content data")
    platform: Platform = Field(description="Source platform")
    author_id: UUID4 = Field(description="Author's user ID")
    author_full_name: str = Field(description="Author's display name")
    
    @property
    def text_content(self) -> str:
        """Extract plain text from content."""
        if isinstance(self.content, dict):
            return self.content.get("text", "")
        return str(self.content)


class ArticleDocument(Document):
    """Long-form article document."""
    
    title: str | None = None
    link: str
    published_date: str | None = None
    tags: list[str] = Field(default_factory=list)
    
    class Settings:
        name = DataCategory.ARTICLES


class PostDocument(Document):
    """Social media post document."""
    
    image_url: str | None = Field(None, alias="image")
    link: str | None = None
    posted_date: str | None = None
    likes: int = 0
    comments: int = 0
    
    class Settings:
        name = DataCategory.POSTS


class RepositoryDocument(Document):
    """Code repository document."""
    
    name: str
    link: str
    description: str | None = None
    language: str | None = None
    stars: int = 0
    forks: int = 0
    
    class Settings:
        name = DataCategory.REPOSITORIES
