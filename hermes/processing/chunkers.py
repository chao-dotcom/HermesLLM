"""Chunkers for splitting documents into chunks."""

from typing import List, Optional

from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter

from hermes.core import CleanedDocument, Chunk
from hermes.processing.base import BaseHandler
from hermes.processing.text_splitters import (
    TokenAwareTextSplitter,
    SentenceAwareSplitter,
    HybridChunker,
)


class TextChunker(BaseHandler):
    """Chunker for splitting text into chunks."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] | None = None
    ) -> None:
        """
        Initialize chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators to split on
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def handle(self, document: CleanedDocument) -> List[Chunk]:
        """
        Split document into chunks.
        
        Args:
            document: Cleaned document
            
        Returns:
            List of chunks
        """
        logger.info(f"Chunking document: {document.id}")
        
        # Split into chunks
        texts = self.splitter.split_text(document.content)
        
        # Create Chunk objects
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                content=text,
                index=i,
                document_id=document.id,
                author_id=document.author_id,
                platform=document.platform,
                metadata={
                    "chunk_size": len(text),
                    "total_chunks": len(texts),
                    "author_name": document.author_full_name
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from document {document.id}")
        return chunks


class TokenAwareChunker(BaseHandler):
    """
    Token-aware chunker that respects embedding model token limits.
    
    This chunker ensures chunks don't exceed the maximum token count,
    preventing truncation during embedding generation. Ideal for use
    with embedding models that have strict token limits.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokens_per_chunk: int = 256,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Initialize token-aware chunker.
        
        Args:
            model_name: HuggingFace model name for tokenizer
            tokens_per_chunk: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens
        """
        self.model_name = model_name
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        
        self.splitter = TokenAwareTextSplitter(
            model_name=model_name,
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
        )
    
    def handle(self, document: CleanedDocument) -> List[Chunk]:
        """
        Split document into token-aware chunks.
        
        Args:
            document: Cleaned document
            
        Returns:
            List of chunks within token limits
        """
        logger.info(f"Token-aware chunking document: {document.id}")
        
        texts = self.splitter.split_text(document.content)
        
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                content=text,
                index=i,
                document_id=document.id,
                author_id=document.author_id,
                platform=document.platform,
                metadata={
                    "chunk_size": len(text),
                    "total_chunks": len(texts),
                    "tokens_per_chunk": self.tokens_per_chunk,
                    "model_name": self.model_name,
                    "author_name": document.author_full_name,
                    "chunking_strategy": "token_aware"
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} token-aware chunks from document {document.id}")
        return chunks


class SentenceAwareChunker(BaseHandler):
    """
    Sentence-aware chunker for articles and blog posts.
    
    This chunker splits text at sentence boundaries, ensuring semantic
    coherence and readability. Ideal for content like articles, blog posts,
    and documentation where preserving complete sentences is important.
    """
    
    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 500,
    ) -> None:
        """
        Initialize sentence-aware chunker.
        
        Args:
            min_length: Minimum chunk length in characters
            max_length: Maximum chunk length in characters
        """
        self.min_length = min_length
        self.max_length = max_length
        
        self.splitter = SentenceAwareSplitter(
            min_length=min_length,
            max_length=max_length,
        )
    
    def handle(self, document: CleanedDocument) -> List[Chunk]:
        """
        Split document into sentence-aware chunks.
        
        Args:
            document: Cleaned document
            
        Returns:
            List of chunks with complete sentences
        """
        logger.info(f"Sentence-aware chunking document: {document.id}")
        
        texts = self.splitter.split_text(document.content)
        
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                content=text,
                index=i,
                document_id=document.id,
                author_id=document.author_id,
                platform=document.platform,
                metadata={
                    "chunk_size": len(text),
                    "total_chunks": len(texts),
                    "min_length": self.min_length,
                    "max_length": self.max_length,
                    "author_name": document.author_full_name,
                    "chunking_strategy": "sentence_aware"
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} sentence-aware chunks from document {document.id}")
        return chunks


class HybridSmartChunker(BaseHandler):
    """
    Hybrid multi-stage chunker combining multiple splitting strategies.
    
    This is the most advanced chunker, using:
    1. Character-based splitting at paragraph boundaries
    2. Token-based splitting to respect model limits
    3. Length filtering for quality control
    
    Recommended for production use with embedding models.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        char_chunk_size: int = 500,
        tokens_per_chunk: int = 256,
        chunk_overlap: int = 50,
        min_chunk_length: int = 50,
        max_chunk_length: int = 1000,
    ) -> None:
        """
        Initialize hybrid smart chunker.
        
        Args:
            model_name: Model name for tokenizer
            char_chunk_size: Character chunk size for first stage
            tokens_per_chunk: Token limit for final chunks
            chunk_overlap: Token overlap between chunks
            min_chunk_length: Minimum chunk length in characters
            max_chunk_length: Maximum chunk length in characters
        """
        self.model_name = model_name
        self.char_chunk_size = char_chunk_size
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        
        self.splitter = HybridChunker(
            char_chunk_size=char_chunk_size,
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
            model_name=model_name,
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
        )
    
    def handle(self, document: CleanedDocument) -> List[Chunk]:
        """
        Split document using hybrid multi-stage approach.
        
        Args:
            document: Cleaned document
            
        Returns:
            List of optimally-sized chunks
        """
        logger.info(f"Hybrid chunking document: {document.id}")
        
        texts = self.splitter.split_text(document.content)
        
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                content=text,
                index=i,
                document_id=document.id,
                author_id=document.author_id,
                platform=document.platform,
                metadata={
                    "chunk_size": len(text),
                    "total_chunks": len(texts),
                    "tokens_per_chunk": self.tokens_per_chunk,
                    "model_name": self.model_name,
                    "author_name": document.author_full_name,
                    "chunking_strategy": "hybrid_smart"
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} hybrid chunks from document {document.id}")
        return chunks


class CodeChunker(BaseHandler):
    """Chunker optimized for code."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> None:
        """
        Initialize code chunker.
        
        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Code-specific separators
        separators = [
            "\n\nclass ",
            "\n\ndef ",
            "\n\nasync def ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def handle(self, document: CleanedDocument) -> List[Chunk]:
        """
        Split code document into chunks.
        
        Args:
            document: Cleaned document
            
        Returns:
            List of chunks
        """
        logger.info(f"Chunking code document: {document.id}")
        
        texts = self.splitter.split_text(document.content)
        
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                content=text,
                index=i,
                document_id=document.id,
                author_id=document.author_id,
                platform=document.platform,
                metadata={
                    "chunk_size": len(text),
                    "total_chunks": len(texts),
                    "type": "code",
                    "author_name": document.author_full_name
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} code chunks from document {document.id}")
        return chunks
