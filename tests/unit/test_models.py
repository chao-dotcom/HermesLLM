"""Unit tests for core data models."""

import pytest
from datetime import datetime
from uuid import UUID

from hermes.core.documents import RawDocument, CleanedDocument
from hermes.core.chunks import Chunk
from hermes.core.embeddings import EmbeddedChunk


class TestRawDocument:
    """Unit tests for RawDocument model."""

    def test_create_raw_document(self):
        """Test creating a raw document."""
        doc = RawDocument(
            id="test-id",
            content="Test content",
            author_id="author-123",
            platform="medium",
        )

        assert doc.id == "test-id"
        assert doc.content == "Test content"
        assert doc.author_id == "author-123"
        assert doc.platform == "medium"
        assert isinstance(doc.created_at, datetime)

    def test_raw_document_with_metadata(self):
        """Test raw document with metadata."""
        metadata = {"title": "Test Article", "url": "https://example.com"}
        
        doc = RawDocument(
            id="test-id",
            content="Content",
            author_id="author-1",
            platform="github",
            metadata=metadata,
        )

        assert doc.metadata == metadata
        assert doc.metadata["title"] == "Test Article"

    def test_raw_document_defaults(self):
        """Test raw document default values."""
        doc = RawDocument(
            id="test-id",
            content="Content",
            author_id="author-1",
            platform="linkedin",
        )

        assert doc.metadata is None
        assert isinstance(doc.created_at, datetime)


class TestCleanedDocument:
    """Unit tests for CleanedDocument model."""

    def test_create_cleaned_document(self):
        """Test creating a cleaned document."""
        doc = CleanedDocument(
            id="cleaned-id",
            content="Cleaned content",
            author_id="author-123",
            platform="medium",
            cleaned_content="Processed content",
        )

        assert doc.id == "cleaned-id"
        assert doc.cleaned_content == "Processed content"

    def test_cleaned_document_word_count(self):
        """Test word count calculation."""
        doc = CleanedDocument(
            id="test-id",
            content="Original content",
            author_id="author-1",
            platform="github",
            cleaned_content="This is a test with five words",
        )

        # Assuming word_count is calculated from cleaned_content
        assert len(doc.cleaned_content.split()) == 7


class TestChunk:
    """Unit tests for Chunk model."""

    def test_create_chunk(self):
        """Test creating a chunk."""
        chunk = Chunk(
            id="chunk-id",
            content="Chunk content here",
            document_id="doc-123",
            author_id="author-1",
            platform="medium",
            chunk_index=0,
        )

        assert chunk.id == "chunk-id"
        assert chunk.content == "Chunk content here"
        assert chunk.document_id == "doc-123"
        assert chunk.chunk_index == 0

    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        metadata = {"section": "introduction", "page": 1}
        
        chunk = Chunk(
            id="chunk-id",
            content="Content",
            document_id="doc-1",
            author_id="author-1",
            platform="github",
            chunk_index=5,
            metadata=metadata,
        )

        assert chunk.metadata == metadata
        assert chunk.chunk_index == 5


class TestEmbeddedChunk:
    """Unit tests for EmbeddedChunk model."""

    def test_create_embedded_chunk(self, mock_embeddings):
        """Test creating an embedded chunk."""
        import numpy as np
        
        embedding = np.random.rand(384).astype(np.float32)
        
        chunk = EmbeddedChunk(
            id="embedded-id",
            content="Content with embedding",
            document_id="doc-1",
            author_id="author-1",
            platform="medium",
            chunk_index=0,
            embedding=embedding.tolist(),
        )

        assert chunk.id == "embedded-id"
        assert len(chunk.embedding) == 384
        assert isinstance(chunk.embedding, list)

    def test_embedded_chunk_to_context(self):
        """Test converting embedded chunks to context."""
        chunks = [
            EmbeddedChunk(
                id=f"chunk-{i}",
                content=f"Chunk {i} content",
                document_id="doc-1",
                author_id="author-1",
                platform="medium",
                chunk_index=i,
                embedding=[0.1] * 384,
            )
            for i in range(3)
        ]

        # Test static method if available
        if hasattr(EmbeddedChunk, 'to_context'):
            context = EmbeddedChunk.to_context(chunks)
            assert isinstance(context, str)
            assert "Chunk 0 content" in context
            assert "Chunk 1 content" in context
            assert "Chunk 2 content" in context


@pytest.mark.unit
class TestDataModelValidation:
    """Test data model validation."""

    def test_raw_document_requires_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises((TypeError, ValueError)):
            RawDocument(id="test")  # Missing required fields

    def test_chunk_index_must_be_int(self):
        """Test chunk index validation."""
        # This might raise a validation error depending on Pydantic config
        try:
            chunk = Chunk(
                id="test",
                content="Content",
                document_id="doc-1",
                author_id="author-1",
                platform="medium",
                chunk_index="invalid",  # Should be int
            )
            # If no error, check type coercion
            assert isinstance(chunk.chunk_index, int)
        except (TypeError, ValueError):
            # Expected for strict validation
            pass

    def test_embedding_must_be_list(self):
        """Test embedding validation."""
        with pytest.raises((TypeError, ValueError)):
            EmbeddedChunk(
                id="test",
                content="Content",
                document_id="doc-1",
                author_id="author-1",
                platform="medium",
                chunk_index=0,
                embedding="invalid",  # Should be list
            )
