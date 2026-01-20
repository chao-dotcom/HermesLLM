"""Medium crawler for articles."""

import requests
from bs4 import BeautifulSoup
from loguru import logger

from hermes.core import ArticleDocument
from hermes.collectors.base import BaseCrawler


class MediumCrawler(BaseCrawler):
    """Crawler for Medium articles."""
    
    model = ArticleDocument
    
    def extract(self, link: str, **kwargs) -> None:
        """
        Extract Medium article data.
        
        Args:
            link: Article URL
            **kwargs: Must contain 'user' key with UserDocument
        """
        # Check if already exists
        old_model = self.model.find_one(link=link)
        if old_model:
            logger.info(f"Article already exists: {link}")
            return
        
        logger.info(f"Scraping Medium article: {link}")
        
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else ""
        
        # Extract subtitle
        subtitle = soup.find("h2")
        subtitle_text = subtitle.get_text(strip=True) if subtitle else ""
        
        # Extract content
        paragraphs = soup.find_all("p")
        content = "\n\n".join([p.get_text(strip=True) for p in paragraphs])
        
        # Save article
        user = kwargs["user"]
        instance = self.model(
            platform="medium",
            link=link,
            content={
                "title": title_text,
                "subtitle": subtitle_text,
                "content": content
            },
            author_id=user.id,
            author_full_name=user.full_name
        )
        instance.save()
        
        logger.info(f"Successfully scraped Medium article: {title_text}")
