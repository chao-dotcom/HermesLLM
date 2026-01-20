"""
Advanced text splitting utilities for token-aware chunking.

This module provides advanced text splitters that respect token limits,
use multi-stage splitting strategies, and intelligently handle sentence boundaries.
"""

import re
from typing import List, Optional, Callable

from loguru import logger
from transformers import AutoTokenizer


class TokenAwareTextSplitter:
    """
    Token-aware text splitter that respects model token limits.
    
    Unlike character-based splitters, this splitter ensures chunks don't exceed
    the maximum token count for embedding models, preventing truncation and
    information loss during embedding generation.
    
    Features:
        - Token-based length measurement
        - Configurable overlap in tokens
        - Support for any HuggingFace tokenizer
        - Smart handling of token boundaries
        
    Example:
        ```python
        splitter = TokenAwareTextSplitter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            tokens_per_chunk=256,
            chunk_overlap=50
        )
        
        text = "Long document text..."
        chunks = splitter.split_text(text)
        ```
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokens_per_chunk: int = 256,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Initialize token-aware splitter.
        
        Args:
            model_name: HuggingFace model name for tokenizer
            tokens_per_chunk: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            
        Raises:
            ValueError: If chunk_overlap >= tokens_per_chunk
        """
        if chunk_overlap >= tokens_per_chunk:
            raise ValueError("chunk_overlap must be less than tokens_per_chunk")
        
        self.model_name = model_name
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"Loading tokenizer for token-aware splitting: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into token-aware chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks, each within token limit
        """
        if not text or not text.strip():
            return []
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.tokens_per_chunk:
            return [text]
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.tokens_per_chunk, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # Move to next chunk with overlap
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - self.chunk_overlap
        
        logger.debug(f"Split text into {len(chunks)} token-aware chunks")
        return chunks


class SentenceAwareSplitter:
    """
    Sentence-aware text splitter that respects sentence boundaries.
    
    This splitter uses regex patterns to identify sentence boundaries and
    creates chunks that end at natural sentence breaks, improving semantic
    coherence and readability.
    
    Features:
        - Smart sentence boundary detection
        - Min/max length enforcement
        - Handles common abbreviations (Dr., Mr., etc.)
        - Preserves sentence integrity
        
    Example:
        ```python
        splitter = SentenceAwareSplitter(
            min_length=100,
            max_length=500
        )
        
        article = "First sentence. Second sentence. Third sentence."
        chunks = splitter.split_text(article)
        ```
    """
    
    # Regex pattern for sentence splitting
    # Matches sentence endings (. ! ?) but handles common abbreviations
    SENTENCE_PATTERN = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
    
    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 500,
        sentence_pattern: Optional[str] = None,
    ) -> None:
        """
        Initialize sentence-aware splitter.
        
        Args:
            min_length: Minimum chunk length in characters
            max_length: Maximum chunk length in characters
            sentence_pattern: Custom regex pattern for sentence splitting
        """
        self.min_length = min_length
        self.max_length = max_length
        self.sentence_pattern = sentence_pattern or self.SENTENCE_PATTERN
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into sentence-aware chunks.
        
        The algorithm builds chunks by adding complete sentences until
        the max_length is reached, ensuring each chunk ends at a sentence
        boundary and meets the min_length requirement.
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks, each containing complete sentences
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences
        sentences = re.split(self.sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding sentence exceeds max length
            if len(current_chunk) + len(sentence) + 1 <= self.max_length:
                current_chunk += sentence + " "
            else:
                # Save current chunk if it meets min length
                if len(current_chunk) >= self.min_length:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    # Current chunk too short, add sentence anyway
                    current_chunk += sentence + " "
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        
        # Add final chunk if it meets min length
        if current_chunk.strip() and len(current_chunk) >= self.min_length:
            chunks.append(current_chunk.strip())
        elif current_chunk.strip() and chunks:
            # Merge short final chunk with last chunk if possible
            chunks[-1] += " " + current_chunk.strip()
        elif current_chunk.strip():
            # Only chunk and it's too short, keep it anyway
            chunks.append(current_chunk.strip())
        
        logger.debug(f"Split text into {len(chunks)} sentence-aware chunks")
        return chunks


class HybridChunker:
    """
    Hybrid multi-stage chunker combining character, sentence, and token-based splitting.
    
    This chunker implements a three-stage strategy:
    1. Character-based splitting to break at paragraph boundaries
    2. Sentence-aware splitting for semantic coherence
    3. Token-based splitting to respect model limits
    
    This approach provides the best balance of semantic coherence and
    token limit compliance, ensuring chunks are both meaningful and
    compatible with embedding models.
    
    Features:
        - Multi-stage splitting pipeline
        - Respects embedding model token limits
        - Maintains semantic coherence
        - Configurable strategies
        
    Example:
        ```python
        from hermes.models import EmbeddingModelSingleton
        
        embedder = EmbeddingModelSingleton()
        chunker = HybridChunker(
            char_chunk_size=500,
            tokens_per_chunk=embedder.max_input_length,
            model_name=embedder.model_id,
            min_chunk_length=50,
            max_chunk_length=1000
        )
        
        text = "Very long document..."
        chunks = chunker.split_text(text)
        ```
    """
    
    def __init__(
        self,
        char_chunk_size: int = 500,
        tokens_per_chunk: int = 256,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_chunk_length: int = 50,
        max_chunk_length: int = 1000,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize hybrid chunker.
        
        Args:
            char_chunk_size: Character chunk size for first stage
            tokens_per_chunk: Token limit for final chunks
            chunk_overlap: Token overlap between chunks
            model_name: Model name for tokenizer
            min_chunk_length: Minimum chunk length in characters
            max_chunk_length: Maximum chunk length in characters
            separators: Custom separators for character splitting
        """
        self.char_chunk_size = char_chunk_size
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        
        # Default separators prioritize paragraph breaks
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        # Initialize token splitter
        self.token_splitter = TokenAwareTextSplitter(
            model_name=model_name,
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
        )
        
        logger.info(f"Initialized HybridChunker with {tokens_per_chunk} tokens per chunk")
    
    def _character_split(self, text: str) -> List[str]:
        """
        Stage 1: Split by characters at paragraph boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of character-based chunks
        """
        if not text:
            return []
        
        chunks = []
        current_chunk = ""
        
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                
                for part in parts:
                    if len(current_chunk) + len(part) + len(separator) <= self.char_chunk_size:
                        current_chunk += part + separator
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = part + separator
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                return chunks
        
        # No separator found, return as is
        return [text] if text else []
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text using multi-stage hybrid approach.
        
        Pipeline:
        1. Character-based splitting at paragraph boundaries
        2. Token-based splitting to respect model limits
        3. Length filtering to enforce min/max constraints
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks optimized for embedding models
        """
        if not text or not text.strip():
            return []
        
        # Stage 1: Character-based splitting
        char_chunks = self._character_split(text)
        logger.debug(f"Stage 1: {len(char_chunks)} character-based chunks")
        
        # Stage 2: Token-based splitting
        token_chunks = []
        for char_chunk in char_chunks:
            token_chunks.extend(self.token_splitter.split_text(char_chunk))
        logger.debug(f"Stage 2: {len(token_chunks)} token-aware chunks")
        
        # Stage 3: Filter by length
        filtered_chunks = []
        for chunk in token_chunks:
            chunk_len = len(chunk)
            if chunk_len >= self.min_chunk_length:
                if chunk_len <= self.max_chunk_length:
                    filtered_chunks.append(chunk)
                else:
                    # Chunk too long, needs further splitting
                    # Split by sentences as a last resort
                    sentence_splitter = SentenceAwareSplitter(
                        min_length=self.min_chunk_length,
                        max_length=self.max_chunk_length
                    )
                    filtered_chunks.extend(sentence_splitter.split_text(chunk))
        
        logger.info(f"Hybrid chunking: {len(filtered_chunks)} final chunks from {len(text)} chars")
        return filtered_chunks


def get_token_count(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> int:
    """
    Get token count for text using specified model's tokenizer.
    
    Args:
        text: Text to count tokens for
        model_name: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def validate_chunk_tokens(
    chunks: List[str],
    max_tokens: int,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> bool:
    """
    Validate that all chunks are within token limit.
    
    Args:
        chunks: List of text chunks to validate
        max_tokens: Maximum allowed tokens per chunk
        model_name: Model name for tokenizer
        
    Returns:
        True if all chunks are within limit, False otherwise
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for i, chunk in enumerate(chunks):
        token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
        if token_count > max_tokens:
            logger.warning(
                f"Chunk {i} exceeds token limit: {token_count} > {max_tokens}"
            )
            return False
    
    return True
