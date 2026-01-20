"""Generate instruction dataset using AI."""

import os
from typing import Dict
from zenml import step
from loguru import logger

from hermes.core import InstructDataset
from hermes.core.enums import DataCategory
from hermes.core.prompts import GenerateDatasetSamplesPrompt
from hermes.datasets.generators import InstructionDatasetGenerator


@step
def generate_instruction_dataset(
    prompts: Dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
    test_split_size: float = 0.2,
    mock: bool = False,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4o-mini",
) -> Dict[str, Dict[DataCategory, InstructDataset]]:
    """
    Generate instruction dataset using LLM.
    
    Args:
        prompts: Dictionary of prompts by category
        test_split_size: Proportion for test set
        mock: Whether to use mocked responses
        openai_api_key: OpenAI API key (or use environment variable)
        openai_model: Model to use for generation
        
    Returns:
        Dictionary with 'train' and 'test' splits
    """
    logger.info(f"Generating instruction dataset (mock={mock})")
    
    # Initialize generator
    generator = InstructionDatasetGenerator(
        openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        openai_model=openai_model
    )
    
    # Generate datasets
    split_datasets = generator.generate(
        prompts=prompts,
        test_size=test_split_size,
        mock=mock
    )
    
    # Log statistics
    train_samples = sum(d.num_samples for d in split_datasets["train"].values())
    test_samples = sum(d.num_samples for d in split_datasets["test"].values())
    
    logger.info(f"Generated {train_samples} training samples and {test_samples} test samples")
    
    return split_datasets
