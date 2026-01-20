"""Custom article crawler for generic websites."""

import requests
from bs4 import BeautifulSoup
from loguru import logger

from hermes.core import ArticleDocument
from hermes.collectors.base import BaseCrawler


class CustomArticleCrawler(BaseCrawler):
    """Generic crawler for custom articles."""
    
    model = ArticleDocument
    
    def extract(self, link: str, **kwargs) -> None:
        """
        Extract custom article data.
        
        Args:
            link: Article URL
            **kwargs: Must contain 'user' key with UserDocument
        """
        # Check if already exists
        old_model = self.model.find_one(link=link)
        if old_model:
            logger.info(f"Article already exists: {link}")
            return
        
        logger.info(f"Scraping custom article: {link}")
        
        try:
            response = requests.get(link, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch {link}: {e}")
            return
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract title
        title = soup.find("h1") or soup.find("title")
        title_text = title.get_text(strip=True) if title else ""
        
        # Extract main content
        # Try to find article/main content
        main_content = (
            soup.find("article") or
            soup.find("main") or
            soup.find("div", class_=["content", "article", "post"])
        )
        
        if main_content:
            paragraphs = main_content.find_all("p")
        else:
            paragraphs = soup.find_all("p")
        
        content = "\n\n".join([p.get_text(strip=True) for p in paragraphs])
        
        # Save article
        user = kwargs["user"]
        instance = self.model(
            platform="custom",
            link=link,
            content={
                "title": title_text,
                "content": content
            },
            author_id=user.id,
            author_full_name=user.full_name
        )
        instance.save()
        
        logger.info(f"Successfully scraped custom article: {title_text}")
