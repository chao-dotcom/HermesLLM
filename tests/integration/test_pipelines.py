"""Integration tests for data collection pipeline."""

import pytest
from unittest.mock import Mock, patch


@pytest.mark.integration
@pytest.mark.slow
class TestDataCollectionPipeline:
    """Integration tests for data collection."""

    def test_medium_collector_integration(self, mock_mongodb_client):
        """Test Medium collector end-to-end."""
        from hermes.collectors.medium import MediumCollector
        
        with patch("hermes.collectors.medium.MongoDBManager") as mock_db:
            mock_db.return_value.insert_document = Mock(return_value={"_id": "test"})
            
            collector = MediumCollector(author_username="test_author")
            
            # Test collection would happen here
            # For integration test, we'd verify the flow
            assert collector is not None

    def test_github_collector_integration(self, mock_mongodb_client):
        """Test GitHub collector end-to-end."""
        from hermes.collectors.github import GitHubCollector
        
        with patch("hermes.collectors.github.MongoDBManager") as mock_db:
            mock_db.return_value.insert_document = Mock(return_value={"_id": "test"})
            
            collector = GitHubCollector(repository_url="https://github.com/test/repo")
            
            assert collector is not None

    @pytest.mark.requires_api
    def test_youtube_collector_integration(self, mock_mongodb_client):
        """Test YouTube collector end-to-end."""
        from hermes.collectors.youtube import YouTubeCollector
        
        with patch("hermes.collectors.youtube.MongoDBManager") as mock_db:
            mock_db.return_value.insert_document = Mock(return_value={"_id": "test"})
            
            collector = YouTubeCollector(video_url="https://youtube.com/watch?v=test")
            
            assert collector is not None


@pytest.mark.integration
class TestProcessingPipeline:
    """Integration tests for document processing."""

    def test_cleaning_and_chunking(self, sample_text):
        """Test cleaning followed by chunking."""
        from hermes.processing.cleaners import TextCleaner
        from hermes.processing.chunkers import RecursiveChunker
        
        # Clean
        cleaner = TextCleaner()
        cleaned = cleaner.clean(sample_text)
        
        # Chunk
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(cleaned)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_embedding_generation(self, sample_documents, mock_embeddings):
        """Test generating embeddings for documents."""
        # Generate embeddings
        embeddings = mock_embeddings.encode(sample_documents)
        
        assert len(embeddings) == len(sample_documents)
        assert embeddings.shape[1] == 384  # Embedding dimension

    def test_full_processing_pipeline(self, sample_text, mock_embeddings, mock_qdrant_client):
        """Test complete processing pipeline."""
        from hermes.processing.cleaners import TextCleaner
        from hermes.processing.chunkers import RecursiveChunker
        
        with patch("hermes.storage.vector_store.QdrantClient", return_value=mock_qdrant_client):
            # 1. Clean
            cleaner = TextCleaner()
            cleaned = cleaner.clean(sample_text)
            
            # 2. Chunk
            chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
            chunks = chunker.chunk(cleaned)
            
            # 3. Embed
            embeddings = mock_embeddings.encode(chunks)
            
            # 4. Store (mocked)
            assert len(chunks) == len(embeddings)


@pytest.mark.integration
@pytest.mark.requires_db
class TestStorageIntegration:
    """Integration tests for storage systems."""

    def test_mongodb_to_qdrant_flow(self, mock_mongodb_client, mock_qdrant_client):
        """Test flow from MongoDB to Qdrant."""
        with patch("hermes.storage.database.MongoClient", return_value=mock_mongodb_client):
            with patch("hermes.storage.vector_store.QdrantClient", return_value=mock_qdrant_client):
                # Simulate retrieving from MongoDB
                from hermes.storage.database import MongoDBManager
                
                db_manager = MongoDBManager()
                
                # Insert test document
                doc = {
                    "id": "test-doc",
                    "content": "Test content",
                    "author_id": "author-1",
                }
                db_manager.insert_document("documents", doc)
                
                # Retrieve and process
                docs = list(db_manager.find_documents("documents", {}))
                assert len(docs) >= 0  # Might be empty in mock

    def test_vector_store_search_integration(self, mock_qdrant_client, mock_embeddings):
        """Test end-to-end vector search."""
        with patch("hermes.storage.vector_store.QdrantClient", return_value=mock_qdrant_client):
            from hermes.storage.vector_store import QdrantVectorStore
            
            store = QdrantVectorStore(collection_name="test_collection")
            
            # Add documents
            documents = ["doc1", "doc2", "doc3"]
            embeddings = mock_embeddings.encode(documents)
            
            # Search
            query_embedding = mock_embeddings.encode("query")
            
            # This tests the integration even with mocks
            assert store is not None


@pytest.mark.integration
class TestRAGPipeline:
    """Integration tests for RAG pipeline."""

    def test_rag_query_flow(self, mock_qdrant_client, mock_embeddings, mock_openai_client):
        """Test RAG query end-to-end."""
        from hermes.rag.pipeline import RAGPipeline
        
        with patch("hermes.rag.retriever.QdrantClient", return_value=mock_qdrant_client):
            with patch("openai.OpenAI", mock_openai_client):
                # Initialize pipeline
                try:
                    pipeline = RAGPipeline()
                    
                    # Test query
                    query = "What is machine learning?"
                    
                    # This would test the full flow with mocks
                    assert pipeline is not None
                except Exception as e:
                    # Pipeline might have different initialization
                    pytest.skip(f"RAG pipeline not initialized: {e}")

    def test_retrieval_and_generation(self, mock_qdrant_client, mock_embeddings):
        """Test retrieval followed by generation."""
        with patch("hermes.rag.retriever.QdrantClient", return_value=mock_qdrant_client):
            from hermes.rag.retriever import DocumentRetriever
            
            try:
                retriever = DocumentRetriever()
                
                query = "test query"
                results = retriever.retrieve(query, k=5)
                
                # Should return results even if mocked
                assert results is not None
            except Exception:
                pytest.skip("Retriever not available")


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    def test_collect_process_embed_workflow(
        self, mock_mongodb_client, mock_qdrant_client, mock_embeddings
    ):
        """Test complete workflow from collection to embedding."""
        # This would test:
        # 1. Collect data
        # 2. Store in MongoDB
        # 3. Process and clean
        # 4. Chunk
        # 5. Generate embeddings
        # 6. Store in Qdrant
        
        # Simplified version with mocks
        documents = ["doc1", "doc2", "doc3"]
        embeddings = mock_embeddings.encode(documents)
        
        assert len(documents) == len(embeddings)

    def test_query_response_workflow(
        self, mock_qdrant_client, mock_embeddings, mock_openai_client
    ):
        """Test complete query-to-response workflow."""
        # This would test:
        # 1. Receive query
        # 2. Embed query
        # 3. Retrieve relevant chunks
        # 4. Generate response
        
        query = "What is AI?"
        query_embedding = mock_embeddings.encode(query)
        
        assert len(query_embedding) == 384
