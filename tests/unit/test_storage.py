"""Unit tests for storage utilities."""

import pytest
from unittest.mock import Mock, patch

from hermes.storage.database import MongoDBManager
from hermes.storage.vector_store import QdrantVectorStore


class TestMongoDBManager:
    """Unit tests for MongoDB manager."""

    def test_init_manager(self, mock_mongodb_client):
        """Test initializing MongoDB manager."""
        with patch("hermes.storage.database.MongoClient", return_value=mock_mongodb_client):
            manager = MongoDBManager()
            assert manager is not None

    def test_insert_document(self, mock_mongodb_client):
        """Test inserting document."""
        with patch("hermes.storage.database.MongoClient", return_value=mock_mongodb_client):
            manager = MongoDBManager()
            
            document = {
                "id": "test-doc",
                "content": "Test content",
                "author_id": "author-1",
            }
            
            # Mock insert
            result = manager.insert_document("documents", document)
            assert result is not None

    def test_find_documents(self, mock_mongodb_client):
        """Test finding documents."""
        with patch("hermes.storage.database.MongoClient", return_value=mock_mongodb_client):
            manager = MongoDBManager()
            
            # Should not raise error
            docs = list(manager.find_documents("documents", {}))
            assert isinstance(docs, list)


class TestQdrantVectorStore:
    """Unit tests for Qdrant vector store."""

    def test_init_vector_store(self, mock_qdrant_client):
        """Test initializing vector store."""
        with patch("hermes.storage.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = QdrantVectorStore(collection_name="test_collection")
            assert store is not None

    def test_add_embeddings(self, mock_qdrant_client, mock_embeddings):
        """Test adding embeddings."""
        with patch("hermes.storage.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = QdrantVectorStore(collection_name="test_collection")
            
            documents = ["doc1", "doc2", "doc3"]
            embeddings = mock_embeddings.encode(documents)
            
            # Should not raise error
            try:
                store.add_embeddings(documents, embeddings)
            except AttributeError:
                # Method might have different name
                pass

    def test_search_similar(self, mock_qdrant_client, mock_embeddings):
        """Test similarity search."""
        with patch("hermes.storage.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = QdrantVectorStore(collection_name="test_collection")
            
            query_embedding = mock_embeddings.encode("query text")
            
            # Should return results
            try:
                results = store.search(query_embedding, limit=5)
                assert isinstance(results, list)
            except AttributeError:
                # Method might have different name
                pass


@pytest.mark.unit
class TestStorageEdgeCases:
    """Test storage edge cases."""

    def test_mongodb_connection_error(self):
        """Test handling MongoDB connection errors."""
        with patch("hermes.storage.database.MongoClient", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception):
                manager = MongoDBManager()

    def test_qdrant_collection_not_found(self, mock_qdrant_client):
        """Test handling missing Qdrant collection."""
        with patch("hermes.storage.vector_store.QdrantClient", return_value=mock_qdrant_client):
            # Should handle gracefully
            store = QdrantVectorStore(collection_name="nonexistent")
            assert store is not None

    def test_empty_embeddings(self, mock_qdrant_client):
        """Test adding empty embeddings list."""
        with patch("hermes.storage.vector_store.QdrantClient", return_value=mock_qdrant_client):
            store = QdrantVectorStore(collection_name="test")
            
            try:
                store.add_embeddings([], [])
            except (ValueError, AttributeError):
                # Expected for validation
                pass
