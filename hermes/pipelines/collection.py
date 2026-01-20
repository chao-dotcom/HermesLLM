"""Data collection pipeline using ZenML."""

from typing import List

from zenml import pipeline, step
from loguru import logger

from hermes.core import UserDocument, ArticleDocument, PostDocument, RepositoryDocument
from hermes.collectors import CrawlerDispatcher


@step
def fetch_user_step(user_id: str, full_name: str) -> UserDocument:
    """
    Fetch or create user document.
    
    Args:
        user_id: User ID
        full_name: User's full name
        
    Returns:
        UserDocument
    """
    logger.info(f"Fetching user: {user_id}")
    
    # Check if user exists
    user = UserDocument.find_one(user_id=user_id)
    
    if not user:
        # Create new user
        user = UserDocument(
            user_id=user_id,
            full_name=full_name
        )
        user.save()
        logger.info(f"Created new user: {user_id}")
    else:
        logger.info(f"Found existing user: {user_id}")
    
    return user


@step
def collect_links_step(
    links: List[str],
    user: UserDocument
) -> Dict[str, int]:
    """
    Collect data from multiple links.
    
    Args:
        links: List of URLs to crawl
        user: User document
        
    Returns:
        Collection statistics
    """
    logger.info(f"Collecting {len(links)} links for user {user.user_id}")
    
    dispatcher = (
        CrawlerDispatcher.build()
        .register_medium()
        .register_github()
        .register_linkedin()
    )
    
    stats = {
        "total": len(links),
        "success": 0,
        "failed": 0
    }
    
    for link in links:
        try:
            crawler = dispatcher.get_crawler(link)
            crawler.extract(link, user=user)
            stats["success"] += 1
            logger.info(f"??Collected: {link}")
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"??Failed to collect {link}: {e}")
    
    logger.info(f"Collection complete: {stats['success']}/{stats['total']} succeeded")
    return stats


@pipeline
def collection_pipeline(
    user_id: str,
    full_name: str,
    links: List[str]
) -> Dict[str, int]:
    """
    Complete data collection pipeline.
    
    Args:
        user_id: User ID
        full_name: User's full name
        links: List of URLs to crawl
        
    Returns:
        Collection statistics
    """
    # Step 1: Get or create user
    user = fetch_user_step(user_id=user_id, full_name=full_name)
    
    # Step 2: Collect all links
    stats = collect_links_step(links=links, user=user)
    
    return stats
