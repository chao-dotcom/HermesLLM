"""
Collection Pipeline

ZenML pipeline for collecting data from various sources.
"""

from typing import List

from zenml import pipeline

from hermes.zenml_steps.collection_steps import (
    create_or_get_author,
    collect_from_links,
)


@pipeline(name="data_collection_pipeline")
def data_collection_pipeline(
    author_full_name: str,
    links: List[str],
    platforms: List[str] = None,
) -> str:
    """
    Pipeline for collecting data from provided links.
    
    Args:
        author_full_name: Full name of the content author
        links: List of URLs to collect from
        platforms: Optional list of platforms to enable
        
    Returns:
        Step invocation ID for downstream dependencies
    """
    # Step 1: Create or get author
    author = create_or_get_author(author_full_name=author_full_name)
    
    # Step 2: Collect from links
    last_step = collect_from_links(
        author=author,
        links=links,
        platforms=platforms,
    )
    
    return last_step.invocation_id
