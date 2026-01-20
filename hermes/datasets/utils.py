"""Utility functions for dataset generation."""

import re
from typing import List

from loguru import logger

from hermes.core.cleaned_documents import CleanedDocument
from hermes.datasets.constants import DEFAULT_MAX_CHUNK_LENGTH, DEFAULT_MIN_CHUNK_LENGTH


def extract_substrings(
    documents: List[CleanedDocument],
    min_length: int = DEFAULT_MIN_CHUNK_LENGTH,
    max_length: int = DEFAULT_MAX_CHUNK_LENGTH,
) -> List[CleanedDocument]:
    """
    Extract meaningful substrings from documents for dataset generation.
    
    This creates smaller, focused excerpts that work better as context
    for generating instruction-answer pairs.
    
    Args:
        documents: List of cleaned documents
        min_length: Minimum substring length
        max_length: Maximum substring length
        
    Returns:
        New list of documents with extracted substrings
    """
    logger.info(f"Extracting substrings from {len(documents)} documents")
    
    extracted_docs = []
    
    for doc in documents:
        content = doc.content
        
        # Skip if document is too short
        if len(content) < min_length:
            logger.debug(f"Skipping document {doc.id} - too short ({len(content)} chars)")
            continue
        
        # If document is within range, use as-is
        if len(content) <= max_length:
            extracted_docs.append(doc)
            continue
        
        # Split into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', content)
        
        # Group sentences into chunks
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed max length
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += sentence + " "
            else:
                # Save current chunk if it meets minimum length
                if len(current_chunk) >= min_length:
                    # Create new document with this chunk
                    chunk_doc = CleanedDocument(
                        content=current_chunk.strip(),
                        platform=doc.platform,
                        author_id=doc.author_id,
                        type=doc.type,
                        metadata={
                            **doc.metadata,
                            "is_substring": True,
                            "parent_id": str(doc.id),
                        }
                    )
                    extracted_docs.append(chunk_doc)
                
                # Start new chunk
                current_chunk = sentence + " "
        
        # Don't forget the last chunk
        if len(current_chunk) >= min_length:
            chunk_doc = CleanedDocument(
                content=current_chunk.strip(),
                platform=doc.platform,
                author_id=doc.author_id,
                type=doc.type,
                metadata={
                    **doc.metadata,
                    "is_substring": True,
                    "parent_id": str(doc.id),
                }
            )
            extracted_docs.append(chunk_doc)
    
    logger.info(f"Extracted {len(extracted_docs)} substrings from {len(documents)} documents")
    return extracted_docs


def batch_items(items: List, batch_size: int) -> List[List]:
    """
    Batch items into smaller groups.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches
