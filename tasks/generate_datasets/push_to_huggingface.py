"""Push generated datasets to HuggingFace Hub."""

import os
from typing import Dict
from zenml import step
from loguru import logger

try:
    from datasets import DatasetDict, concatenate_datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("HuggingFace datasets not installed")

from hermes.core import InstructDataset, PreferenceDataset
from hermes.core.enums import DataCategory


@step
def push_to_huggingface(
    datasets: Dict[str, Dict[DataCategory, InstructDataset | PreferenceDataset]],
    dataset_id: str,
    huggingface_token: str | None = None,
    flatten: bool = True,
) -> None:
    """
    Push generated datasets to HuggingFace Hub.
    
    Args:
        datasets: Dictionary with 'train' and 'test' splits
        dataset_id: HuggingFace dataset ID (e.g., 'username/dataset-name')
        huggingface_token: HuggingFace API token
        flatten: Whether to flatten categories into single train/test
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("HuggingFace datasets library required. Install with: pip install datasets")
    
    logger.info(f"Pushing datasets to HuggingFace Hub: {dataset_id}")
    
    token = huggingface_token or os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required. Set HUGGINGFACE_ACCESS_TOKEN environment variable.")
    
    # Convert to HuggingFace format
    train_datasets = []
    test_datasets = []
    
    for category, dataset in datasets["train"].items():
        train_datasets.append(dataset.to_huggingface())
        logger.info(f"Train {category}: {dataset.num_samples} samples")
    
    for category, dataset in datasets["test"].items():
        test_datasets.append(dataset.to_huggingface())
        logger.info(f"Test {category}: {dataset.num_samples} samples")
    
    # Combine or keep separate
    if flatten:
        train_data = concatenate_datasets(train_datasets)
        test_data = concatenate_datasets(test_datasets)
    else:
        # Keep categories separate (would need different structure)
        logger.warning("Non-flattened mode not fully implemented, using flatten=True")
        train_data = concatenate_datasets(train_datasets)
        test_data = concatenate_datasets(test_datasets)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": train_data,
        "test": test_data,
    })
    
    # Push to Hub
    dataset_dict.push_to_hub(
        dataset_id,
        token=token,
        private=False,
    )
    
    logger.info(f"Successfully pushed {len(train_data)} train and {len(test_data)} test samples to {dataset_id}")
