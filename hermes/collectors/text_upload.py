"""Text upload collector for direct text submission."""

from loguru import logger

from hermes.core import ArticleDocument
from hermes.collectors.base import BaseCrawler


class TextUploadCrawler(BaseCrawler):
    """Collector for direct text uploads."""
    
    model = ArticleDocument
    
    def extract(self, link: str, **kwargs) -> None:
        """
        Extract data from direct text upload.
        
        Args:
            link: Unique identifier/URL for the uploaded text
            **kwargs: Must contain:
                - 'user': UserDocument
                - 'title': str - Title of the content
                - 'content': str - Main text content
                - 'description': str (optional) - Brief description
                - 'language': str (optional) - Content language (default: 'en')
                - 'tags': list[str] (optional) - Content tags
        """
        # Check if already exists
        old_model = self.model.find_one(link=link)
        if old_model:
            logger.info(f"Text upload already exists: {link}")
            return
        
        # Extract required parameters
        user = kwargs.get("user")
        if not user:
            logger.error("User parameter is required for text upload")
            return
        
        title = kwargs.get("title", "")
        content = kwargs.get("content", "")
        description = kwargs.get("description", "")
        language = kwargs.get("language", "en")
        tags = kwargs.get("tags", [])
        
        # Validate required fields
        if not title:
            logger.error("Title is required for text upload")
            return
        
        if not content:
            logger.error("Content is required for text upload")
            return
        
        logger.info(f"Processing text upload: {title}")
        
        # Build content structure
        content_data = {
            "title": title,
            "content": content,
            "language": language
        }
        
        if description:
            content_data["description"] = description
        
        # Save article
        instance = self.model(
            platform="text_upload",
            link=link,
            title=title,
            content=content_data,
            author_id=user.id,
            author_full_name=user.full_name,
            tags=tags
        )
        instance.save()
        
        logger.info(f"Successfully saved text upload: {title}")
    
    @classmethod
    def upload_text(
        cls,
        user,
        title: str,
        content: str,
        link: str | None = None,
        description: str | None = None,
        language: str = "en",
        tags: list[str] | None = None
    ) -> ArticleDocument | None:
        """
        Convenience method for uploading text directly.
        
        Args:
            user: UserDocument instance
            title: Content title
            content: Main text content
            link: Unique identifier (auto-generated if None)
            description: Brief description
            language: Content language
            tags: Content tags
            
        Returns:
            ArticleDocument instance or None if failed
        """
        import uuid
        
        # Auto-generate link if not provided
        if not link:
            link = f"text_upload://{uuid.uuid4()}"
        
        crawler = cls()
        crawler.extract(
            link=link,
            user=user,
            title=title,
            content=content,
            description=description or "",
            language=language,
            tags=tags or []
        )
        
        # Return the saved document
        return cls.model.find_one(link=link)
