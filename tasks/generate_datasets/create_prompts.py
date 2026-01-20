"""Create prompts from cleaned documents."""

from typing import Dict, List
from zenml import step
from loguru import logger

from hermes.core import CleanedDocument, DatasetType
from hermes.core.enums import DataCategory
from hermes.core.prompts import GenerateDatasetSamplesPrompt
from hermes.datasets.generators import InstructionDatasetGenerator, PreferenceDatasetGenerator


@step
def create_prompts(
    documents: List[CleanedDocument],
    dataset_type: DatasetType = DatasetType.INSTRUCTION,
) -> Dict[DataCategory, List[GenerateDatasetSamplesPrompt]]:
    """
    Create generation prompts from cleaned documents.
    
    Args:
        documents: List of cleaned documents
        dataset_type: Type of dataset to generate
        
    Returns:
        Dictionary mapping categories to lists of prompts
    """
    logger.info(f"Creating {dataset_type.value} prompts from {len(documents)} documents")
    
    # Select appropriate generator
    if dataset_type == DatasetType.INSTRUCTION:
        generator_class = InstructionDatasetGenerator
    elif dataset_type == DatasetType.PREFERENCE:
        generator_class = PreferenceDatasetGenerator
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # Generate prompts
    prompts = generator_class.get_prompts(documents, extract_chunks=True)
    
    total_prompts = sum(len(p) for p in prompts.values())
    logger.info(f"Created {total_prompts} prompts across {len(prompts)} categories")
    
    return prompts
