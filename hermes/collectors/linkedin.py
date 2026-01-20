"""LinkedIn crawler for profile and posts."""

import time
from typing import Dict, List

from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger

try:
    from selenium.webdriver.common.by import By
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from hermes.config import get_settings
from hermes.core import PostDocument
from hermes.core.exceptions import ImproperlyConfigured
from hermes.collectors.base import BaseSeleniumCrawler


class LinkedInCrawler(BaseSeleniumCrawler):
    """
    LinkedIn crawler for profiles and posts.
    
    Note: LinkedIn has updated security measures, so this crawler
    may not work as expected. It's marked as deprecated.
    """
    
    model = PostDocument
    
    def __init__(self, scroll_limit: int = 5, is_deprecated: bool = True) -> None:
        super().__init__(scroll_limit)
        self._is_deprecated = is_deprecated
    
    def set_extra_driver_options(self, options) -> None:
        """Add LinkedIn-specific options."""
        options.add_experimental_option("detach", True)
    
    def login(self) -> None:
        """Login to LinkedIn."""
        if self._is_deprecated:
            raise DeprecationWarning(
                "LinkedIn login is deprecated due to updated security measures"
            )
        
        settings = get_settings()
        
        self.driver.get("https://www.linkedin.com/login")
        
        if not settings.linkedin_username or not settings.linkedin_password:
            raise ImproperlyConfigured(
                "LinkedIn credentials required: linkedin_username and linkedin_password"
            )
        
        self.driver.find_element(By.ID, "username").send_keys(settings.linkedin_username)
        self.driver.find_element(By.ID, "password").send_keys(settings.linkedin_password)
        self.driver.find_element(By.CSS_SELECTOR, ".login__form_action_container button").click()
    
    def extract(self, link: str, **kwargs) -> None:
        """
        Extract LinkedIn profile data.
        
        Args:
            link: Profile URL
            **kwargs: Must contain 'user' key with UserDocument
        """
        if self._is_deprecated:
            raise DeprecationWarning(
                "LinkedIn extraction is deprecated due to updated feed structure"
            )
        
        # Check if already exists
        old_model = self.model.find_one(link=link)
        if old_model:
            logger.info(f"Posts already exist for: {link}")
            return
        
        logger.info(f"Scraping LinkedIn profile: {link}")
        
        self.login()
        
        # Get profile content
        soup = self._get_page_content(link)
        
        # Extract profile sections
        data = {
            "Name": self._scrape_section(soup, "h1", class_="text-heading-xlarge"),
            "About": self._scrape_section(soup, "div", class_="display-flex ph5 pv3"),
            "Main Page": self._scrape_section(soup, "div", {"id": "main-content"}),
            "Experience": self._scrape_experience(link),
            "Education": self._scrape_education(link),
        }
        
        # Navigate to posts
        self.driver.get(link)
        time.sleep(5)
        
        try:
            button = self.driver.find_element(
                By.CSS_SELECTOR,
                ".app-aware-link.profile-creator-shared-content-view__footer-action"
            )
            button.click()
        except Exception as e:
            logger.warning(f"Could not click posts button: {e}")
        
        # Scroll and scrape posts
        self.scroll_page()
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        
        post_elements = soup.find_all(
            "div",
            class_="update-components-text relative update-components-update-v2__commentary"
        )
        buttons = soup.find_all("button", class_="update-components-image__image-link")
        post_images = self._extract_image_urls(buttons)
        
        posts = self._extract_posts(post_elements, post_images)
        logger.info(f"Found {len(posts)} posts")
        
        self.driver.close()
        
        # Save posts
        user = kwargs["user"]
        PostDocument.bulk_insert([
            PostDocument(
                platform="linkedin",
                content=post,
                author_id=user.id,
                author_full_name=user.full_name
            )
            for post in posts.values()
        ])
        
        logger.info(f"Finished scraping: {link}")
    
    def _scrape_section(self, soup: BeautifulSoup, *args, **kwargs) -> str:
        """Scrape a specific section."""
        parent_div = soup.find(*args, **kwargs)
        return parent_div.get_text(strip=True) if parent_div else ""
    
    def _extract_image_urls(self, buttons: List[Tag]) -> Dict[str, str]:
        """Extract image URLs from buttons."""
        post_images = {}
        for i, button in enumerate(buttons):
            img_tag = button.find("img")
            if img_tag and "src" in img_tag.attrs:
                post_images[f"Post_{i}"] = img_tag["src"]
            else:
                logger.warning(f"No image in button {i}")
        return post_images
    
    def _get_page_content(self, url: str) -> BeautifulSoup:
        """Get page content as BeautifulSoup."""
        self.driver.get(url)
        time.sleep(5)
        return BeautifulSoup(self.driver.page_source, "html.parser")
    
    def _extract_posts(
        self,
        post_elements: List[Tag],
        post_images: Dict[str, str]
    ) -> Dict[str, Dict[str, str]]:
        """Extract post data."""
        posts_data = {}
        for i, post_element in enumerate(post_elements):
            post_text = post_element.get_text(strip=True)
            post_key = f"Post_{i}"
            posts_data[post_key] = {"text": post_text}
            
            if post_key in post_images:
                posts_data[post_key]["image"] = post_images[post_key]
        
        return posts_data
    
    def _scrape_experience(self, profile_url: str) -> str:
        """Scrape experience section."""
        exp_url = f"{profile_url}/details/experience"
        soup = self._get_page_content(exp_url)
        return self._scrape_section(soup, "div", {"id": "experience"})
    
    def _scrape_education(self, profile_url: str) -> str:
        """Scrape education section."""
        edu_url = f"{profile_url}/details/education"
        soup = self._get_page_content(edu_url)
        return self._scrape_section(soup, "div", {"id": "education"})
