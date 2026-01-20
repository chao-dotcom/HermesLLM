"""Data collectors for web scraping and data ingestion."""

from atlas.collectors.base import BaseCrawler, BaseSeleniumCrawler
from atlas.collectors.dispatcher import CrawlerDispatcher
from atlas.collectors.linkedin import LinkedInCrawler
from atlas.collectors.medium import MediumCrawler
from atlas.collectors.github import GithubCrawler
from atlas.collectors.custom_article import CustomArticleCrawler
from atlas.collectors.text_upload import TextUploadCrawler
from atlas.collectors.youtube import YouTubeTranscriptCrawler

__all__ = [
    "BaseCrawler",
    "BaseSeleniumCrawler",
    "CrawlerDispatcher",
    "LinkedInCrawler",
    "MediumCrawler",
    "GithubCrawler",
    "CustomArticleCrawler",
    "TextUploadCrawler",
    "YouTubeTranscriptCrawler",
]
