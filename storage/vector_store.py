"""Qdrant vector database connection and operations."""

from functools import lru_cache
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from hermes.config import get_settings


class QdrantStore:
    """Qdrant vector database manager."""
    
    _instance: QdrantClient | None = None
    
    def __new__(cls) -> "QdrantStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._client: QdrantClient | None = None
    
    def connect(
        self,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None
    ) -> QdrantClient:
        """
        Connect to Qdrant vector database.
        
        Args:
            url: Cloud URL (for Qdrant Cloud)
            host: Local host (for self-hosted)
            port: Local port (for self-hosted)
            api_key: API key (for Qdrant Cloud)
            
        Returns:
            Qdrant client instance
        """
        if self._client is not None:
            return self._client
        
        settings = get_settings()
        
        try:
            if settings.use_qdrant_cloud:
                url = url or settings.qdrant_cloud_url
                api_key = api_key or settings.qdrant_api_key
                
                self._client = QdrantClient(url=url, api_key=api_key)
                logger.info(f"Connected to Qdrant Cloud: {url}")
            else:
                host = host or settings.qdrant_host
                port = port or settings.qdrant_port
                
                self._client = QdrantClient(host=host, port=port)
                logger.info(f"Connected to Qdrant: {host}:{port}")
            
            return self._client
            
        except UnexpectedResponse as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def get_client(self) -> QdrantClient:
        """Get Qdrant client (connects if needed)."""
        if self._client is None:
            return self.connect()
        return self._client
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of collection
            vector_size: Dimension of vectors
            distance: Distance metric
            
        Returns:
            True if created successfully
        """
        try:
            client = self.get_client()
            
            # Check if collection exists
            collections = client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            
            logger.info(f"Created collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False
    
    def upsert_points(
        self,
        collection_name: str,
        points: list[PointStruct]
    ) -> bool:
        """
        Insert or update points in collection.
        
        Args:
            collection_name: Name of collection
            points: List of points to upsert
            
        Returns:
            True if successful
        """
        try:
            client = self.get_client()
            client.upsert(collection_name=collection_name, points=points)
            logger.debug(f"Upserted {len(points)} points to '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filters: Filter | None = None,
        score_threshold: float | None = None
    ) -> list[Any]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Name of collection
            query_vector: Query embedding
            limit: Max results to return
            filters: Optional metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        try:
            client = self.get_client()
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filters,
                score_threshold=score_threshold
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            client = self.get_client()
            client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def close(self) -> None:
        """Close connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Qdrant connection closed")


@lru_cache()
def get_vector_store() -> QdrantStore:
    """Get Qdrant store instance (cached)."""
    store = QdrantStore()
    store.connect()
    return store
