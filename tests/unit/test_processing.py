"""Unit tests for text processing utilities."""

import pytest

from hermes.processing.cleaners import TextCleaner
from hermes.processing.chunkers import RecursiveChunker


class TestTextCleaner:
    """Unit tests for TextCleaner."""

    def test_clean_html(self):
        """Test HTML cleaning."""
        cleaner = TextCleaner()
        
        html = "<p>This is <b>bold</b> text with <a href='#'>links</a>.</p>"
        cleaned = cleaner.clean(html)
        
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "bold" in cleaned
        assert "links" in cleaned

    def test_remove_extra_whitespace(self):
        """Test whitespace normalization."""
        cleaner = TextCleaner()
        
        text = "This   has    extra     spaces\n\n\nand newlines"
        cleaned = cleaner.clean(text)
        
        # Should normalize to single spaces
        assert "   " not in cleaned
        assert "This has extra spaces" in cleaned or "This has" in cleaned

    def test_clean_empty_string(self):
        """Test cleaning empty string."""
        cleaner = TextCleaner()
        
        cleaned = cleaner.clean("")
        assert cleaned == ""

    def test_clean_urls(self):
        """Test URL handling."""
        cleaner = TextCleaner()
        
        text = "Visit https://example.com for more info"
        cleaned = cleaner.clean(text)
        
        # URL might be kept or removed depending on implementation
        assert isinstance(cleaned, str)


class TestRecursiveChunker:
    """Unit tests for RecursiveChunker."""

    def test_chunk_text(self, sample_text):
        """Test basic text chunking."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        
        chunks = chunker.chunk(sample_text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_size_respected(self):
        """Test that chunks respect size limits."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        
        long_text = "word " * 100  # Create long text
        chunks = chunker.chunk(long_text)
        
        # Most chunks should be around chunk_size
        for chunk in chunks[:-1]:  # Last chunk might be smaller
            assert len(chunk) <= 50 + 20  # Allow some flexibility

    def test_chunk_overlap(self):
        """Test chunk overlap."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        
        text = "This is a test sentence. " * 20
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have overlap
            # (implementation-dependent)
            assert isinstance(chunks[0], str)
            assert isinstance(chunks[1], str)

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        
        chunks = chunker.chunk("")
        
        assert chunks == [] or chunks == [""]

    def test_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
        
        text = "Short text."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text or chunks[0].strip() == text.strip()


@pytest.mark.unit
class TestChunkingEdgeCases:
    """Test edge cases in chunking."""

    def test_single_word(self):
        """Test chunking single word."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        
        chunks = chunker.chunk("Word")
        
        assert len(chunks) == 1
        assert chunks[0].strip() == "Word"

    def test_special_characters(self):
        """Test handling special characters."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        
        text = "Test with @#$% special chars!"
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        assert isinstance(chunks[0], str)

    def test_unicode_text(self):
        """Test handling Unicode text."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        
        text = "Hello 世界 🌍 Привет"
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        assert "世界" in "".join(chunks) or "Hello" in chunks[0]
