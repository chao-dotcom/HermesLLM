"""
Advanced chunking utilities and helpers.

This module provides utility functions and helpers for working with
text chunks, including validation, analysis, and optimization.
"""

from typing import List, Dict, Any, Tuple
import statistics

from loguru import logger
from transformers import AutoTokenizer

from hermes.core import Chunk


def analyze_chunks(
    chunks: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Analyze chunks to provide statistics and insights.
    
    Args:
        chunks: List of text chunks to analyze
        model_name: Model name for token counting
        
    Returns:
        Dictionary with chunk statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "total_tokens": 0,
            "avg_chars": 0,
            "avg_tokens": 0,
            "min_chars": 0,
            "max_chars": 0,
            "min_tokens": 0,
            "max_tokens": 0,
        }
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    char_lengths = [len(chunk) for chunk in chunks]
    token_lengths = [
        len(tokenizer.encode(chunk, add_special_tokens=False))
        for chunk in chunks
    ]
    
    return {
        "total_chunks": len(chunks),
        "total_characters": sum(char_lengths),
        "total_tokens": sum(token_lengths),
        "avg_chars": statistics.mean(char_lengths),
        "avg_tokens": statistics.mean(token_lengths),
        "median_chars": statistics.median(char_lengths),
        "median_tokens": statistics.median(token_lengths),
        "min_chars": min(char_lengths),
        "max_chars": max(char_lengths),
        "min_tokens": min(token_lengths),
        "max_tokens": max(token_lengths),
        "std_chars": statistics.stdev(char_lengths) if len(char_lengths) > 1 else 0,
        "std_tokens": statistics.stdev(token_lengths) if len(token_lengths) > 1 else 0,
    }


def validate_chunk_quality(
    chunks: List[str],
    min_length: int = 50,
    max_length: int = 1000,
    max_tokens: int = 512,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[bool, List[str]]:
    """
    Validate chunk quality against multiple criteria.
    
    Args:
        chunks: List of chunks to validate
        min_length: Minimum character length
        max_length: Maximum character length
        max_tokens: Maximum token count
        model_name: Model name for tokenizer
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        
        # Check character length
        if chunk_len < min_length:
            issues.append(f"Chunk {i}: Too short ({chunk_len} < {min_length} chars)")
        if chunk_len > max_length:
            issues.append(f"Chunk {i}: Too long ({chunk_len} > {max_length} chars)")
        
        # Check token count
        token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
        if token_count > max_tokens:
            issues.append(f"Chunk {i}: Too many tokens ({token_count} > {max_tokens})")
        
        # Check for empty or whitespace-only chunks
        if not chunk.strip():
            issues.append(f"Chunk {i}: Empty or whitespace-only")
    
    return len(issues) == 0, issues


def merge_short_chunks(
    chunks: List[str],
    min_length: int = 100,
    max_length: int = 1000,
) -> List[str]:
    """
    Merge chunks that are too short with adjacent chunks.
    
    Args:
        chunks: List of chunks to process
        min_length: Minimum desired chunk length
        max_length: Maximum allowed chunk length
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
    
    merged = []
    current = chunks[0]
    
    for i in range(1, len(chunks)):
        if len(current) < min_length and len(current) + len(chunks[i]) <= max_length:
            # Merge with next chunk
            current += " " + chunks[i]
        else:
            merged.append(current)
            current = chunks[i]
    
    # Add final chunk
    if current:
        if merged and len(current) < min_length and len(merged[-1]) + len(current) <= max_length:
            # Merge final short chunk with previous
            merged[-1] += " " + current
        else:
            merged.append(current)
    
    logger.info(f"Merged {len(chunks)} chunks into {len(merged)} chunks")
    return merged


def split_long_chunks(
    chunks: List[str],
    max_length: int = 1000,
    separator: str = " ",
) -> List[str]:
    """
    Split chunks that are too long into smaller chunks.
    
    Args:
        chunks: List of chunks to process
        max_length: Maximum allowed chunk length
        separator: Separator to use for splitting
        
    Returns:
        List of split chunks
    """
    result = []
    
    for chunk in chunks:
        if len(chunk) <= max_length:
            result.append(chunk)
        else:
            # Split on separator
            parts = chunk.split(separator)
            current = ""
            
            for part in parts:
                if len(current) + len(part) + len(separator) <= max_length:
                    current += part + separator
                else:
                    if current:
                        result.append(current.rstrip())
                    current = part + separator
            
            if current:
                result.append(current.rstrip())
    
    logger.info(f"Split {len(chunks)} chunks into {len(result)} chunks")
    return result


def optimize_chunks(
    chunks: List[str],
    min_length: int = 100,
    max_length: int = 1000,
    max_tokens: int = 512,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[str]:
    """
    Optimize chunks by merging short ones and splitting long ones.
    
    Args:
        chunks: List of chunks to optimize
        min_length: Minimum chunk length
        max_length: Maximum chunk length
        max_tokens: Maximum token count
        model_name: Model name for validation
        
    Returns:
        List of optimized chunks
    """
    logger.info(f"Optimizing {len(chunks)} chunks")
    
    # First pass: merge short chunks
    merged = merge_short_chunks(chunks, min_length, max_length)
    
    # Second pass: split long chunks
    optimized = split_long_chunks(merged, max_length)
    
    # Validate
    is_valid, issues = validate_chunk_quality(
        optimized,
        min_length=min_length,
        max_length=max_length,
        max_tokens=max_tokens,
        model_name=model_name,
    )
    
    if not is_valid:
        logger.warning(f"Optimized chunks have {len(issues)} issues")
        for issue in issues[:5]:  # Show first 5 issues
            logger.warning(f"  - {issue}")
    
    logger.success(f"Optimized to {len(optimized)} chunks")
    return optimized


def get_chunk_overlap_stats(chunks: List[str]) -> Dict[str, Any]:
    """
    Analyze overlap between consecutive chunks.
    
    Args:
        chunks: List of chunks to analyze
        
    Returns:
        Dictionary with overlap statistics
    """
    if len(chunks) < 2:
        return {"avg_overlap_chars": 0, "avg_overlap_ratio": 0}
    
    overlaps = []
    
    for i in range(len(chunks) - 1):
        current = chunks[i]
        next_chunk = chunks[i + 1]
        
        # Find common suffix/prefix
        overlap_len = 0
        max_overlap = min(len(current), len(next_chunk))
        
        for j in range(1, max_overlap + 1):
            if current[-j:] == next_chunk[:j]:
                overlap_len = j
        
        overlaps.append(overlap_len)
    
    avg_overlap = statistics.mean(overlaps)
    avg_ratio = avg_overlap / statistics.mean([len(c) for c in chunks])
    
    return {
        "avg_overlap_chars": avg_overlap,
        "avg_overlap_ratio": avg_ratio,
        "min_overlap": min(overlaps),
        "max_overlap": max(overlaps),
    }


def filter_chunks_by_content(
    chunks: List[Chunk],
    min_word_count: int = 10,
    exclude_patterns: List[str] = None,
) -> List[Chunk]:
    """
    Filter chunks based on content quality.
    
    Args:
        chunks: List of Chunk objects
        min_word_count: Minimum number of words
        exclude_patterns: Patterns to exclude (e.g., ["404", "error"])
        
    Returns:
        Filtered list of chunks
    """
    exclude_patterns = exclude_patterns or []
    filtered = []
    
    for chunk in chunks:
        # Check word count
        word_count = len(chunk.content.split())
        if word_count < min_word_count:
            logger.debug(f"Filtered out chunk {chunk.index}: too few words ({word_count})")
            continue
        
        # Check exclude patterns
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern.lower() in chunk.content.lower():
                logger.debug(f"Filtered out chunk {chunk.index}: contains '{pattern}'")
                should_exclude = True
                break
        
        if not should_exclude:
            filtered.append(chunk)
    
    logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} chunks")
    return filtered


def deduplicate_chunks(chunks: List[Chunk], similarity_threshold: float = 0.95) -> List[Chunk]:
    """
    Remove duplicate or near-duplicate chunks.
    
    Uses simple character-based similarity. For semantic deduplication,
    use embedding-based methods instead.
    
    Args:
        chunks: List of Chunk objects
        similarity_threshold: Similarity threshold (0-1) for considering duplicates
        
    Returns:
        Deduplicated list of chunks
    """
    if not chunks:
        return []
    
    unique_chunks = [chunks[0]]
    
    for chunk in chunks[1:]:
        is_duplicate = False
        
        for unique in unique_chunks:
            # Simple character-based similarity
            similarity = _calculate_similarity(chunk.content, unique.content)
            
            if similarity >= similarity_threshold:
                logger.debug(
                    f"Filtered duplicate chunk {chunk.index}: "
                    f"{similarity:.2%} similar to chunk {unique.index}"
                )
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
    return unique_chunks


def _calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple character-based similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    if text1 == text2:
        return 1.0
    
    # Use set intersection of words as simple similarity metric
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)
