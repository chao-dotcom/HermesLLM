"""RAG retrieval system."""

from typing import List

from loguru import logger

from hermes.core import EmbeddedQuery
from hermes.core.chunks import Chunk
from hermes.storage.vector_store import QdrantStore


class VectorRetriever:
    """Vector-based retriever using Qdrant."""
    
    def __init__(
        self,
        collection_name: str = "chunks",
        top_k: int = 5
    ) -> None:
        """
        Initialize retriever.
        
        Args:
            collection_name: Name of Qdrant collection
            top_k: Number of results to retrieve
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.vector_store = QdrantStore()
    
    def retrieve(self, query: EmbeddedQuery) -> List[Chunk]:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: Embedded query
            
        Returns:
            List of relevant chunks
        """
        logger.info(f"Retrieving top {self.top_k} chunks for query")
        
        # Search vector store
        results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query.embedding,
            limit=self.top_k
        )
        
        # Convert results to Chunk objects
        chunks = []
        for result in results:
            chunk = Chunk(
                id=result.id,
                content=result.payload.get("content", ""),
                index=result.payload.get("index", 0),
                document_id=result.payload.get("document_id"),
                author_id=result.payload.get("author_id"),
                platform=result.payload.get("platform", ""),
                metadata=result.payload.get("metadata", {})
            )
            chunks.append(chunk)
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks


class HybridRetriever:
    """Hybrid retriever combining vector and keyword search."""
    
    def __init__(
        self,
        collection_name: str = "chunks",
        top_k: int = 5,
        vector_weight: float = 0.7
    ) -> None:
        """
        Initialize hybrid retriever.
        
        Args:
            collection_name: Name of Qdrant collection
            top_k: Number of results to retrieve
            vector_weight: Weight for vector search (1 - weight for keyword)
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.keyword_weight = 1.0 - vector_weight
        self.vector_store = QdrantStore()
    
    def retrieve(self, query: EmbeddedQuery) -> List[Chunk]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Embedded query
            
        Returns:
            List of relevant chunks
        """
        logger.info(f"Hybrid retrieval for query with weights: vector={self.vector_weight}, keyword={self.keyword_weight}")
        
        # Vector search
        vector_results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query.embedding,
            limit=self.top_k * 2  # Get more results for fusion
        )
        
        # TODO: Add keyword search when implemented in Qdrant
        # For now, just use vector search
        
        # Take top K results
        results = vector_results[:self.top_k]
        
        # Convert to Chunk objects
        chunks = []
        for result in results:
            chunk = Chunk(
                id=result.id,
                content=result.payload.get("content", ""),
                index=result.payload.get("index", 0),
                document_id=result.payload.get("document_id"),
                author_id=result.payload.get("author_id"),
                platform=result.payload.get("platform", ""),
                metadata=result.payload.get("metadata", {})
            )
            chunks.append(chunk)
        
        logger.info(f"Retrieved {len(chunks)} chunks via hybrid search")
        return chunks
