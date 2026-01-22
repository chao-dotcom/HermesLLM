"""Example unit test to verify test infrastructure."""

import pytest


def test_basic_assertion():
    """Test basic assertion."""
    assert True


def test_string_equality():
    """Test string equality."""
    string = "unit_test_example"
    assert string == "unit_test_example"


def test_list_operations():
    """Test list operations."""
    items = [1, 2, 3, 4, 5]
    
    assert len(items) == 5
    assert 3 in items
    assert items[0] == 1


def test_dict_operations():
    """Test dictionary operations."""
    data = {"key1": "value1", "key2": "value2"}
    
    assert "key1" in data
    assert data["key1"] == "value1"
    assert len(data) == 2


@pytest.mark.unit
def test_with_marker():
    """Test with unit marker."""
    result = 2 + 2
    assert result == 4


def test_fixture_usage(sample_text):
    """Test using a fixture."""
    assert isinstance(sample_text, str)
    assert len(sample_text) > 0
    assert "machine learning" in sample_text.lower()


def test_multiple_fixtures(sample_documents, sample_metadata):
    """Test using multiple fixtures."""
    assert isinstance(sample_documents, list)
    assert len(sample_documents) > 0
    
    assert isinstance(sample_metadata, dict)
    assert "author" in sample_metadata
    assert sample_metadata["author"] == "test_author"


class TestExampleClass:
    """Example test class."""

    def test_method_one(self):
        """Test method one."""
        assert 1 + 1 == 2

    def test_method_two(self):
        """Test method two."""
        text = "test"
        assert text.upper() == "TEST"

    def test_with_fixture(self, sample_text):
        """Test with fixture in class."""
        assert "machine" in sample_text.lower()


@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (10, 20),
])
def test_parametrized(input, expected):
    """Test parametrized test."""
    result = input * 2
    assert result == expected


@pytest.mark.parametrize("text,contains", [
    ("hello world", "world"),
    ("python programming", "python"),
    ("machine learning", "learning"),
])
def test_string_contains(text, contains):
    """Test string contains."""
    assert contains in text
