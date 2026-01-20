"""Base crawler classes for data collection."""

import time
from abc import ABC, abstractmethod
from tempfile import mkdtemp

from loguru import logger

try:
    import chromedriver_autoinstaller
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    
    SELENIUM_AVAILABLE = True
    # Auto-install chromedriver if needed
    chromedriver_autoinstaller.install()
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not installed. Install with: pip install selenium webdriver-manager")

from hermes.core.base import MongoDocument


class BaseCrawler(ABC):
    """Base class for all crawlers."""
    
    model: type[MongoDocument]
    
    @abstractmethod
    def extract(self, link: str, **kwargs) -> None:
        """
        Extract data from URL.
        
        Args:
            link: URL to crawl
            **kwargs: Additional arguments (user, etc.)
        """
        pass


class BaseSeleniumCrawler(BaseCrawler, ABC):
    """Base class for Selenium-based crawlers."""
    
    def __init__(self, scroll_limit: int = 5) -> None:
        """
        Initialize Selenium crawler.
        
        Args:
            scroll_limit: Maximum number of scrolls
        """
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required for this crawler")
        
        options = webdriver.ChromeOptions()
        
        # Headless options
        options.add_argument("--no-sandbox")
        options.add_argument("--headless=new")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--ignore-certificate-errors")
        
        # Temporary directories
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        options.add_argument("--remote-debugging-port=9226")
        
        # Allow subclasses to add options
        self.set_extra_driver_options(options)
        
        self.scroll_limit = scroll_limit
        self.driver = webdriver.Chrome(options=options)
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def set_extra_driver_options(self, options: Options) -> None:
        """
        Override to add custom driver options.
        
        Args:
            options: Chrome options to modify
        """
        pass
    
    def login(self) -> None:
        """Override to implement login logic."""
        pass
    
    def scroll_page(self) -> None:
        """Scroll through page based on scroll limit."""
        current_scroll = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height or (self.scroll_limit and current_scroll >= self.scroll_limit):
                break
            
            last_height = new_height
            current_scroll += 1
        
        logger.debug(f"Scrolled {current_scroll} times")
    
    def __del__(self):
        """Cleanup driver on deletion."""
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
            except Exception:
                pass
