"""Dataset generation pipeline."""

from zenml import pipeline
from loguru import logger

from hermes.core import DatasetType
from tasks.generate_datasets import (
    query_cleaned_documents,
    create_prompts,
    generate_instruction_dataset,
    generate_preference_dataset,
    push_to_huggingface,
)


@pipeline
def generate_datasets_pipeline(
    dataset_type: str = "instruction",
    test_split_size: float = 0.2,
    push_to_hub: bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
    limit: int | None = None,
) -> None:
    """
    Pipeline to generate training datasets from cleaned documents using AI.
    
    This pipeline:
    1. Queries cleaned documents from the database
    2. Creates prompts for dataset generation
    3. Uses LLM to generate instruction or preference datasets
    4. Optionally pushes datasets to HuggingFace Hub
    
    Args:
        dataset_type: Type of dataset ('instruction' or 'preference')
        test_split_size: Proportion of data for test set
        push_to_hub: Whether to push to HuggingFace Hub
        dataset_id: HuggingFace dataset ID (required if push_to_hub=True)
        mock: Whether to use mocked LLM responses for testing
        limit: Limit number of documents to process (for testing)
    """
    logger.info(f"Starting dataset generation pipeline: {dataset_type}")
    
    # Convert string to enum
    if dataset_type == "instruction":
        ds_type = DatasetType.INSTRUCTION
    elif dataset_type == "preference":
        ds_type = DatasetType.PREFERENCE
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    
    # Step 1: Query cleaned documents
    documents = query_cleaned_documents(limit=limit)
    
    # Step 2: Create prompts
    prompts = create_prompts(documents=documents, dataset_type=ds_type)
    
    # Step 3: Generate datasets
    if ds_type == DatasetType.INSTRUCTION:
        datasets = generate_instruction_dataset(
            prompts=prompts,
            test_split_size=test_split_size,
            mock=mock,
        )
    else:  # PREFERENCE
        datasets = generate_preference_dataset(
            prompts=prompts,
            test_split_size=test_split_size,
            mock=mock,
        )
    
    # Step 4: Optionally push to HuggingFace
    if push_to_hub:
        if not dataset_id:
            raise ValueError("dataset_id required when push_to_hub=True")
        push_to_huggingface(
            datasets=datasets,
            dataset_id=dataset_id,
            flatten=True,
        )
    
    logger.info("Dataset generation pipeline completed")
