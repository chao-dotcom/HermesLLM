"""Crawler dispatcher for routing URLs to appropriate crawlers."""

import re
from urllib.parse import urlparse

from loguru import logger

from atlas.collectors.base import BaseCrawler


class CrawlerDispatcher:
    """Dispatcher for routing URLs to appropriate crawlers."""
    
    def __init__(self) -> None:
        self._crawlers: dict[str, type[BaseCrawler]] = {}
    
    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        """Build dispatcher with default crawlers registered."""
        dispatcher = cls()
        return dispatcher
    
    def register_medium(self) -> "CrawlerDispatcher":
        """Register Medium crawler."""
        from atlas.collectors.medium import MediumCrawler
        self.register("https://medium.com", MediumCrawler)
        return self
    
    def register_linkedin(self) -> "CrawlerDispatcher":
        """Register LinkedIn crawler."""
        from atlas.collectors.linkedin import LinkedInCrawler
        self.register("https://linkedin.com", LinkedInCrawler)
        return self
    
    def register_github(self) -> "CrawlerDispatcher":
        """Register GitHub crawler."""
        from atlas.collectors.github import GithubCrawler
        self.register("https://github.com", GithubCrawler)
        return self
    
    def register_youtube(self) -> "CrawlerDispatcher":
        """Register YouTube crawler."""
        from atlas.collectors.youtube import YouTubeTranscriptCrawler
        self.register("https://youtube.com", YouTubeTranscriptCrawler)
        self.register("https://youtu.be", YouTubeTranscriptCrawler)
        return self
    
    def register_text_upload(self) -> "CrawlerDispatcher":
        """Register Text Upload crawler."""
        from atlas.collectors.text_upload import TextUploadCrawler
        self.register("text_upload://", TextUploadCrawler)
        return self
    
    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        """
        Register a crawler for a domain.
        
        Args:
            domain: Domain URL
            crawler: Crawler class
        """
        parsed_domain = urlparse(domain)
        domain = parsed_domain.netloc
        
        pattern = r"https://(www\.)?{}/*".format(re.escape(domain))
        self._crawlers[pattern] = crawler
        
        logger.info(f"Registered {crawler.__name__} for pattern: {pattern}")
    
    def get_crawler(self, url: str) -> BaseCrawler:
        """
        Get appropriate crawler for URL.
        
        Args:
            url: URL to crawl
            
        Returns:
            Crawler instance
        """
        for pattern, crawler_class in self._crawlers.items():
            if re.match(pattern, url):
                logger.debug(f"Matched {crawler_class.__name__} for {url}")
                return crawler_class()
        
        # Default to custom article crawler
        logger.warning(f"No crawler found for {url}. Using CustomArticleCrawler")
        from atlas.collectors.custom_article import CustomArticleCrawler
        return CustomArticleCrawler()
