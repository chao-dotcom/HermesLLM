"""Prompt models for LLM interactions."""

from atlas.core.base import VectorDocument
from atlas.core.cleaned_documents import CleanedDocument
from atlas.core.enums import DataCategory


class Prompt(VectorDocument):
    """Prompt template and content."""
    
    template: str
    input_variables: dict
    content: str
    num_tokens: int | None = None
    
    class Config:
        category = DataCategory.PROMPTS


class GenerateDatasetSamplesPrompt(Prompt):
    """Prompt for generating dataset samples."""
    
    data_category: DataCategory
    document: CleanedDocument
