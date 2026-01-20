"""
Collection (ETL) Steps for ZenML Pipelines

This module contains ZenML steps for data collection and ETL operations.
"""

from typing import List, Dict, Any
from urllib.parse import urlparse

from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step

from hermes.core import Document
from hermes.collectors.dispatcher import CollectorDispatcher


@step
def create_or_get_author(author_full_name: str) -> Annotated[Dict[str, Any], "author"]:
    """
    Create or retrieve author information.
    
    Args:
        author_full_name: Full name of the author
        
    Returns:
        Author information dictionary
    """
    logger.info(f"Processing author: {author_full_name}")
    
    # Parse name
    name_parts = author_full_name.split()
    first_name = name_parts[0] if name_parts else ""
    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
    
    author_info = {
        "full_name": author_full_name,
        "first_name": first_name,
        "last_name": last_name,
        "author_id": author_full_name.lower().replace(" ", "_"),
    }
    
    logger.success(f"Author ready: {author_info['author_id']}")
    return author_info


@step
def collect_from_links(
    author: Dict[str, Any],
    links: List[str],
    platforms: List[str] = None,
) -> Annotated[List[str], "collected_links"]:
    """
    Collect data from provided links using appropriate collectors.
    
    Args:
        author: Author information dictionary
        links: List of URLs to collect from
        platforms: Optional list of platforms to enable
        
    Returns:
        List of successfully collected links
    """
    platforms = platforms or ["medium", "github", "linkedin", "youtube"]
    
    # Build dispatcher with registered collectors
    dispatcher = CollectorDispatcher()
    for platform in platforms:
        try:
            dispatcher.register(platform)
        except ValueError as e:
            logger.warning(f"Could not register {platform}: {e}")
    
    logger.info(f"Starting to collect from {len(links)} link(s) for {author['full_name']}")
    
    successful_collections = []
    failed_collections = []
    metadata = {}
    
    for link in tqdm(links, desc="Collecting"):
        try:
            # Get appropriate collector for link
            collector = dispatcher.get_collector(link)
            domain = urlparse(link).netloc
            
            # Collect data
            documents = collector.collect(
                link=link,
                author_full_name=author["full_name"],
                author_id=author["author_id"]
            )
            
            successful_collections.append(link)
            _update_metadata(metadata, domain, success=True)
            
            logger.debug(f"Collected {len(documents)} documents from {link}")
            
        except Exception as e:
            logger.error(f"Failed to collect from {link}: {e}")
            failed_collections.append(link)
            try:
                domain = urlparse(link).netloc
                _update_metadata(metadata, domain, success=False)
            except:
                pass
    
    # Add step metadata
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="collected_links",
        metadata={
            **metadata,
            "total_links": len(links),
            "successful": len(successful_collections),
            "failed": len(failed_collections),
            "success_rate": len(successful_collections) / len(links) if links else 0,
        }
    )
    
    logger.success(
        f"Collection complete: {len(successful_collections)}/{len(links)} successful"
    )
    
    return successful_collections


@step
def collect_from_platform(
    author: Dict[str, Any],
    platform: str,
    query: str = None,
    max_items: int = 10,
) -> Annotated[int, "documents_collected"]:
    """
    Collect data from a specific platform using search/API.
    
    Args:
        author: Author information dictionary
        platform: Platform name (e.g., "medium", "github")
        query: Optional search query
        max_items: Maximum number of items to collect
        
    Returns:
        Number of documents collected
    """
    logger.info(f"Collecting from {platform} for {author['full_name']}")
    
    try:
        dispatcher = CollectorDispatcher()
        dispatcher.register(platform)
        
        # For platform-wide collection, use author name as query if not provided
        search_query = query or author["full_name"]
        
        # Get collector and collect
        collector = dispatcher.get_collector_by_platform(platform)
        
        # Note: This is a simplified version - actual implementation
        # would depend on platform-specific APIs
        logger.info(f"Searching {platform} with query: '{search_query}'")
        
        # Placeholder for actual collection logic
        # documents = collector.search_and_collect(
        #     query=search_query,
        #     author_id=author["author_id"],
        #     max_items=max_items
        # )
        
        documents_count = 0  # Placeholder
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="documents_collected",
            metadata={
                "platform": platform,
                "query": search_query,
                "max_items": max_items,
                "collected": documents_count,
            }
        )
        
        logger.success(f"Collected {documents_count} documents from {platform}")
        return documents_count
        
    except Exception as e:
        logger.error(f"Failed to collect from {platform}: {e}")
        return 0


def _update_metadata(
    metadata: Dict[str, Dict[str, int]],
    domain: str,
    success: bool
) -> None:
    """Update metadata dictionary with collection results."""
    if domain not in metadata:
        metadata[domain] = {"successful": 0, "failed": 0, "total": 0}
    
    metadata[domain]["total"] += 1
    if success:
        metadata[domain]["successful"] += 1
    else:
        metadata[domain]["failed"] += 1
