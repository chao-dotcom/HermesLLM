"""
Dataset Generation Pipeline

ZenML pipeline for generating instruction and preference datasets.
"""

from typing import Optional

from zenml import pipeline

from hermes.core.enums import DatasetType
from hermes.zenml_steps.dataset_steps import (
    query_cleaned_documents,
    generate_instruction_dataset,
    generate_preference_dataset,
    push_to_huggingface,
)


@pipeline(name="dataset_generation_pipeline")
def dataset_generation_pipeline(
    dataset_type: DatasetType = DatasetType.INSTRUCTION,
    num_samples: int = 100,
    test_split_size: float = 0.1,
    model: str = "gpt-4o-mini",
    push_to_hf: bool = False,
    dataset_id: Optional[str] = None,
    mock: bool = False,
    wait_for: Optional[str] = None,
) -> Optional[str]:
    """
    Pipeline for generating instruction or preference datasets.
    
    This pipeline:
    1. Queries cleaned documents from database
    2. Generates dataset (instruction or preference) using LLM
    3. Splits into train/test sets
    4. Optionally pushes to HuggingFace Hub
    
    Args:
        dataset_type: Type of dataset (INSTRUCTION or PREFERENCE)
        num_samples: Number of samples to generate
        test_split_size: Fraction of data for test set (0-1)
        model: OpenAI model to use for generation
        push_to_hf: Whether to push to HuggingFace Hub
        dataset_id: HuggingFace dataset ID (required if push_to_hf=True)
        mock: Whether to use mock generation (for testing)
        wait_for: Optional step ID to wait for
        
    Returns:
        Optional push status if pushing to HuggingFace
    """
    # Step 1: Query cleaned documents
    cleaned_documents = query_cleaned_documents(wait_for=wait_for)
    
    # Step 2: Generate dataset based on type
    if dataset_type == DatasetType.INSTRUCTION:
        train_dataset, test_dataset = generate_instruction_dataset(
            cleaned_documents=cleaned_documents,
            test_split_size=test_split_size,
            num_samples=num_samples,
            model=model,
            mock=mock,
        )
    elif dataset_type == DatasetType.PREFERENCE:
        train_dataset, test_dataset = generate_preference_dataset(
            cleaned_documents=cleaned_documents,
            test_split_size=test_split_size,
            num_samples=num_samples,
            model=model,
            mock=mock,
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    
    # Step 3: Optionally push to HuggingFace
    if push_to_hf:
        if not dataset_id:
            raise ValueError("dataset_id is required when push_to_hf=True")
        
        push_status = push_to_huggingface(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            dataset_id=dataset_id,
        )
        return push_status.invocation_id
    
    return None
