"""Cleaners for different document types."""

import re
from typing import Dict, Any

from loguru import logger

from hermes.core import ArticleDocument, PostDocument, RepositoryDocument, CleanedDocument
from hermes.processing.base import BaseHandler


class ArticleCleaner(BaseHandler):
    """Cleaner for article documents."""
    
    def handle(self, article: ArticleDocument) -> CleanedDocument:
        """
        Clean article document.
        
        Args:
            article: Article to clean
            
        Returns:
            Cleaned document
        """
        logger.info(f"Cleaning article: {article.id}")
        
        content = article.content or {}
        
        # Extract and clean title
        title = content.get("title", "")
        title = self._clean_text(title)
        
        # Extract and clean content
        text = content.get("content", "")
        text = self._clean_text(text)
        
        # Combine cleaned content
        cleaned_content = f"# {title}\n\n{text}" if title else text
        
        return CleanedDocument(
            content=cleaned_content,
            platform=article.platform,
            document_id=article.id,
            author_id=article.author_id,
            author_full_name=article.author_full_name,
            type="article"
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove special characters (keep letters, numbers, punctuation)
        text = re.sub(r'[^\w\s.,!?;:\'"-]', '', text)
        return text.strip()


class PostCleaner(BaseHandler):
    """Cleaner for post documents."""
    
    def handle(self, post: PostDocument) -> CleanedDocument:
        """
        Clean post document.
        
        Args:
            post: Post to clean
            
        Returns:
            Cleaned document
        """
        logger.info(f"Cleaning post: {post.id}")
        
        content = post.content or ""
        
        # Clean text
        cleaned_content = self._clean_text(content)
        
        return CleanedDocument(
            content=cleaned_content,
            platform=post.platform,
            document_id=post.id,
            author_id=post.author_id,
            author_full_name=post.author_full_name,
            type="post"
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove mentions and hashtags (keep the text)
        text = re.sub(r'[@#](\w+)', r'\1', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?;:\'"-]', '', text)
        return text.strip()


class RepositoryCleaner(BaseHandler):
    """Cleaner for repository documents."""
    
    def handle(self, repo: RepositoryDocument) -> CleanedDocument:
        """
        Clean repository document.
        
        Args:
            repo: Repository to clean
            
        Returns:
            Cleaned document
        """
        logger.info(f"Cleaning repository: {repo.id}")
        
        content = repo.content or {}
        
        # Extract README content
        files = content.get("files", {})
        readme = files.get("README.md", "") or files.get("README.rst", "") or files.get("README.txt", "")
        
        # Extract tree structure
        tree = content.get("tree", "")
        
        # Combine content
        cleaned_parts = []
        
        if repo.name:
            cleaned_parts.append(f"# Repository: {repo.name}")
        
        if tree:
            cleaned_parts.append(f"## File Structure\n{tree}")
        
        if readme:
            cleaned_parts.append(f"## README\n{readme}")
        
        # Add other important files
        for filename, file_content in files.items():
            if filename not in ["README.md", "README.rst", "README.txt"] and file_content:
                cleaned_parts.append(f"## {filename}\n{file_content}")
        
        cleaned_content = "\n\n".join(cleaned_parts)
        
        return CleanedDocument(
            content=cleaned_content,
            platform=repo.platform,
            document_id=repo.id,
            author_id=repo.author_id,
            author_full_name=repo.author_full_name,
            type="repository"
        )
