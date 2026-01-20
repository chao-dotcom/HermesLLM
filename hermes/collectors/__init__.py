"""Data collectors for web scraping and data ingestion."""

from hermes.collectors.base import BaseCrawler, BaseSeleniumCrawler
from hermes.collectors.dispatcher import CrawlerDispatcher
from hermes.collectors.linkedin import LinkedInCrawler
from hermes.collectors.medium import MediumCrawler
from hermes.collectors.github import GithubCrawler
from hermes.collectors.custom_article import CustomArticleCrawler
from hermes.collectors.text_upload import TextUploadCrawler
from hermes.collectors.youtube import YouTubeTranscriptCrawler

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
