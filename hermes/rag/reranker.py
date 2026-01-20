"""Reranker for retrieved chunks."""

from typing import List

from loguru import logger

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False

from hermes.core import Query
from hermes.core.chunks import Chunk
from hermes.models import CrossEncoderModelSingleton


class CrossEncoderReranker:
    """Reranker using cross-encoder models with singleton pattern."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 3,
        device: str = "cpu",
        use_singleton: bool = True,
    ) -> None:
        """
        Initialize reranker.
        
        Args:
            model_name: Name of cross-encoder model
            top_k: Number of top chunks to keep after reranking
            device: Device to run model on ('cpu', 'cuda', 'mps')
            use_singleton: Whether to use singleton pattern (recommended)
        """
        if not CROSSENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self.use_singleton = use_singleton
        
        if use_singleton:
            logger.info(f"Using CrossEncoderModelSingleton for {model_name}")
            self.model = CrossEncoderModelSingleton(model_id=model_name, device=device)
        else:
            logger.info(f"Loading reranker model: {model_name}")
            self.model = CrossEncoder(model_name, device=device)
    
    def rerank(self, query: Query, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rerank chunks based on relevance to query.
        
        Args:
            query: Query object
            chunks: List of chunks to rerank
            
        Returns:
            Reranked list of top K chunks
        """
        if not chunks:
            return []
        
        logger.info(f"Reranking {len(chunks)} chunks")
        
        # Prepare pairs for cross-encoder
        pairs = [(query.content, chunk.content) for chunk in chunks]
        
        # Get relevance scores
        if self.use_singleton:
            scores = self.model(pairs, to_list=False)
        else:
            scores = self.model.predict(pairs)
        
        # Sort by score
        ranked_indices = scores.argsort()[::-1]
        
        # Return top K
        reranked_chunks = [chunks[i] for i in ranked_indices[:self.top_k]]
        
        logger.info(f"Reranked to top {len(reranked_chunks)} chunks")
        return reranked_chunks


class SimpleReranker:
    """Simple reranker based on keyword matching."""
    
    def __init__(self, top_k: int = 3) -> None:
        """
        Initialize simple reranker.
        
        Args:
            top_k: Number of top chunks to keep
        """
        self.top_k = top_k
    
    def rerank(self, query: Query, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rerank chunks based on keyword overlap.
        
        Args:
            query: Query object
            chunks: List of chunks to rerank
            
        Returns:
            Reranked list of top K chunks
        """
        if not chunks:
            return []
        
        logger.info(f"Simple reranking of {len(chunks)} chunks")
        
        # Tokenize query
        query_tokens = set(query.content.lower().split())
        
        # Score each chunk
        scores = []
        for chunk in chunks:
            chunk_tokens = set(chunk.content.lower().split())
            overlap = len(query_tokens & chunk_tokens)
            scores.append(overlap)
        
        # Sort by score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        # Return top K
        reranked_chunks = [chunks[i] for i in ranked_indices[:self.top_k]]
        
        logger.info(f"Reranked to top {len(reranked_chunks)} chunks")
        return reranked_chunks
