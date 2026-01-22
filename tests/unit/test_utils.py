"""Unit tests for utility functions."""

import pytest
from datetime import datetime

from hermes.utils.helpers import generate_unique_id, format_timestamp, clean_filename


class TestHelpers:
    """Unit tests for helper utilities."""

    def test_generate_unique_id(self):
        """Test unique ID generation."""
        id1 = generate_unique_id()
        id2 = generate_unique_id()
        
        assert id1 != id2
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        formatted = format_timestamp(dt)
        
        assert isinstance(formatted, str)
        assert "2024" in formatted

    def test_clean_filename(self):
        """Test filename cleaning."""
        dirty = "test/file:name*.txt"
        clean = clean_filename(dirty)
        
        # Should remove invalid characters
        assert "/" not in clean
        assert ":" not in clean
        assert "*" not in clean

    def test_clean_filename_unicode(self):
        """Test cleaning Unicode filenames."""
        filename = "test_文件名.txt"
        clean = clean_filename(filename)
        
        assert isinstance(clean, str)
        # Should handle Unicode gracefully


@pytest.mark.unit
class TestUtilityEdgeCases:
    """Test utility edge cases."""

    def test_empty_filename(self):
        """Test cleaning empty filename."""
        try:
            result = clean_filename("")
            assert result == "" or result == "unnamed"
        except ValueError:
            # Expected for validation
            pass

    def test_none_timestamp(self):
        """Test formatting None timestamp."""
        try:
            result = format_timestamp(None)
            assert result is not None
        except (TypeError, AttributeError):
            # Expected for validation
            pass
