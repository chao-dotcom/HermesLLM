"""Query cleaned documents for dataset generation."""

from typing import List
from zenml import step
from loguru import logger

from hermes.core import CleanedDocument


@step
def query_cleaned_documents(
    limit: int | None = None,
    author_full_names: List[str] | None = None,
) -> List[CleanedDocument]:
    """
    Query cleaned documents from the database.
    
    Args:
        limit: Maximum number of documents to retrieve
        author_full_names: Filter by author names (optional)
        
    Returns:
        List of cleaned documents
    """
    logger.info(f"Querying cleaned documents (limit={limit}, authors={author_full_names})")
    
    query_filter = {}
    
    # Add author filter if specified
    if author_full_names:
        # This would need to resolve author names to IDs
        # For now, just log it
        logger.info(f"Filtering by authors: {author_full_names}")
    
    # Query documents from database
    documents = list(CleanedDocument.find(query_filter, limit=limit))
    
    logger.info(f"Retrieved {len(documents)} cleaned documents")
    
    return documents
