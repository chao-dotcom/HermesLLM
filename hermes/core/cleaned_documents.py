"""Cleaned document models for processed content."""

from abc import ABC

from pydantic import UUID4

from hermes.core.base import VectorDocument
from hermes.core.enums import DataCategory, Platform


class CleanedDocument(VectorDocument, ABC):
    """Base class for cleaned documents."""
    
    content: str
    platform: Platform
    author_id: UUID4
    author_full_name: str


class CleanedPostDocument(CleanedDocument):
    """Cleaned social media post."""
    
    image_url: str | None = None
    
    class Config:
        name = "cleaned_posts"
        category = DataCategory.POSTS
        use_vector_index = False


class CleanedArticleDocument(CleanedDocument):
    """Cleaned article document."""
    
    link: str
    
    class Config:
        name = "cleaned_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


class CleanedRepositoryDocument(CleanedDocument):
    """Cleaned repository document."""
    
    name: str
    link: str
    
    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False
