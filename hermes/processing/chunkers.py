"""Chunkers for splitting documents into chunks."""

from typing import List

from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter

from hermes.core import CleanedDocument, Chunk
from hermes.processing.base import BaseHandler


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
