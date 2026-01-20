"""Embedders for generating vector embeddings."""

from typing import List

from loguru import logger
from sentence_transformers import SentenceTransformer

from hermes.core import Chunk, EmbeddedChunk
from hermes.models import EmbeddingModelSingleton
from hermes.processing.base import BaseHandler


class SentenceTransformerEmbedder(BaseHandler):
    """Embedder using Sentence Transformers with singleton pattern."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        use_singleton: bool = True,
    ) -> None:
        """
        Initialize embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run model on ('cpu', 'cuda', 'mps')
            use_singleton: Whether to use singleton pattern (recommended)
        """
        self.model_name = model_name
        self.device = device
        self.use_singleton = use_singleton
        
        if use_singleton:
            logger.info(f"Using EmbeddingModelSingleton for {model_name}")
            self.model = EmbeddingModelSingleton(model_id=model_name, device=device)
        else:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
    
    def handle(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of embedded chunks
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batch
        if self.use_singleton:
            embeddings = self.model(
                texts,
                to_list=False,
                show_progress_bar=True,
                normalize_embeddings=True
            )
        else:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True
            )
        
        # Create embedded chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = EmbeddedChunk(
                content=chunk.content,
                embedding=embedding.tolist(),
                index=chunk.index,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                author_id=chunk.author_id,
                platform=chunk.platform,
                metadata={
                    **chunk.metadata,
                    "model": self.model_name,
                    "embedding_dim": len(embedding)
                }
            )
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks


class InstructorEmbedder(BaseHandler):
    """Embedder using Instructor models for asymmetric search."""
    
    def __init__(
        self,
        model_name: str = "hkunlp/instructor-base",
        instruction: str = "Represent the document for retrieval:"
    ) -> None:
        """
        Initialize instructor embedder.
        
        Args:
            model_name: Name of the instructor model
            instruction: Instruction prefix for embeddings
        """
        try:
            from InstructorEmbedding import INSTRUCTOR
            self.model_name = model_name
            self.instruction = instruction
            logger.info(f"Loading instructor model: {model_name}")
            self.model = INSTRUCTOR(model_name)
        except ImportError:
            raise ImportError(
                "InstructorEmbedding not installed. "
                "Install with: pip install InstructorEmbedding"
            )
    
    def handle(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of embedded chunks
        """
        logger.info(f"Generating instructor embeddings for {len(chunks)} chunks")
        
        # Prepare texts with instructions
        texts = [[self.instruction, chunk.content] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create embedded chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = EmbeddedChunk(
                content=chunk.content,
                embedding=embedding.tolist(),
                index=chunk.index,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                author_id=chunk.author_id,
                platform=chunk.platform,
                metadata={
                    **chunk.metadata,
                    "model": self.model_name,
                    "instruction": self.instruction,
                    "embedding_dim": len(embedding)
                }
            )
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"Generated instructor embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
