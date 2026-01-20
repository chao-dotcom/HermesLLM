"""
Dataset Generation Steps for ZenML Pipelines

This module contains steps for generating instruction and preference datasets.
"""

from typing import List, Tuple
from typing_extensions import Annotated

from loguru import logger
from zenml import get_step_context, step

from hermes.core import CleanedDocument, Dataset
from hermes.core.enums import DatasetType
from hermes.storage.database import DatabaseClient
from hermes.training.dataset_builder import DatasetBuilder


@step
def query_cleaned_documents(
    author_names: List[str] = None,
    limit: int = None,
    wait_for: str = None,
) -> Annotated[List[CleanedDocument], "cleaned_documents"]:
    """
    Query cleaned documents from database for dataset generation.
    
    Args:
        author_names: Optional list of author names
        limit: Optional limit on documents
        wait_for: Optional step ID to wait for
        
    Returns:
        List of cleaned documents
    """
    logger.info("Querying cleaned documents for dataset generation")
    
    try:
        db_client = DatabaseClient()
        
        query_filter = {}
        if author_names:
            query_filter["author_full_name"] = {"$in": author_names}
        
        documents = db_client.find_cleaned_documents(
            filter=query_filter,
            limit=limit
        )
        
        logger.success(f"Retrieved {len(documents)} cleaned documents")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="cleaned_documents",
            metadata={
                "total_documents": len(documents),
                "authors": author_names or "all",
            }
        )
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to query cleaned documents: {e}")
        return []


@step
def generate_instruction_dataset(
    cleaned_documents: List[CleanedDocument],
    test_split_size: float = 0.1,
    num_samples: int = 100,
    model: str = "gpt-4o-mini",
    mock: bool = False,
) -> Annotated[Tuple[Dataset, Dataset], "instruction_datasets"]:
    """
    Generate instruction dataset from cleaned documents.
    
    Args:
        cleaned_documents: List of cleaned documents
        test_split_size: Fraction of data for test set
        num_samples: Number of samples to generate
        model: OpenAI model to use
        mock: Whether to use mock data (for testing)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    logger.info(f"Generating instruction dataset from {len(cleaned_documents)} documents")
    
    try:
        builder = DatasetBuilder(
            dataset_type=DatasetType.INSTRUCTION,
            model=model,
        )
        
        if mock:
            logger.warning("Using mock dataset generation")
            # Create mock datasets for testing
            train_dataset = Dataset(
                dataset_type=DatasetType.INSTRUCTION,
                samples=[],
                metadata={"mock": True, "split": "train"}
            )
            test_dataset = Dataset(
                dataset_type=DatasetType.INSTRUCTION,
                samples=[],
                metadata={"mock": True, "split": "test"}
            )
        else:
            # Generate real dataset
            dataset = builder.generate_from_documents(
                documents=cleaned_documents,
                num_samples=num_samples,
            )
            
            # Split into train/test
            train_dataset, test_dataset = builder.split_dataset(
                dataset=dataset,
                test_size=test_split_size,
            )
        
        logger.success(
            f"Generated instruction dataset: "
            f"{len(train_dataset.samples)} train, {len(test_dataset.samples)} test"
        )
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="instruction_datasets",
            metadata={
                "dataset_type": "instruction",
                "train_samples": len(train_dataset.samples),
                "test_samples": len(test_dataset.samples),
                "test_split_size": test_split_size,
                "mock": mock,
            }
        )
        
        return train_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Failed to generate instruction dataset: {e}")
        # Return empty datasets on error
        empty_train = Dataset(dataset_type=DatasetType.INSTRUCTION, samples=[])
        empty_test = Dataset(dataset_type=DatasetType.INSTRUCTION, samples=[])
        return empty_train, empty_test


@step
def generate_preference_dataset(
    cleaned_documents: List[CleanedDocument],
    test_split_size: float = 0.1,
    num_samples: int = 100,
    model: str = "gpt-4o-mini",
    mock: bool = False,
) -> Annotated[Tuple[Dataset, Dataset], "preference_datasets"]:
    """
    Generate preference dataset from cleaned documents.
    
    Args:
        cleaned_documents: List of cleaned documents
        test_split_size: Fraction of data for test set
        num_samples: Number of samples to generate
        model: OpenAI model to use
        mock: Whether to use mock data (for testing)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    logger.info(f"Generating preference dataset from {len(cleaned_documents)} documents")
    
    try:
        builder = DatasetBuilder(
            dataset_type=DatasetType.PREFERENCE,
            model=model,
        )
        
        if mock:
            logger.warning("Using mock dataset generation")
            train_dataset = Dataset(
                dataset_type=DatasetType.PREFERENCE,
                samples=[],
                metadata={"mock": True, "split": "train"}
            )
            test_dataset = Dataset(
                dataset_type=DatasetType.PREFERENCE,
                samples=[],
                metadata={"mock": True, "split": "test"}
            )
        else:
            dataset = builder.generate_from_documents(
                documents=cleaned_documents,
                num_samples=num_samples,
            )
            
            train_dataset, test_dataset = builder.split_dataset(
                dataset=dataset,
                test_size=test_split_size,
            )
        
        logger.success(
            f"Generated preference dataset: "
            f"{len(train_dataset.samples)} train, {len(test_dataset.samples)} test"
        )
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="preference_datasets",
            metadata={
                "dataset_type": "preference",
                "train_samples": len(train_dataset.samples),
                "test_samples": len(test_dataset.samples),
                "test_split_size": test_split_size,
                "mock": mock,
            }
        )
        
        return train_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Failed to generate preference dataset: {e}")
        empty_train = Dataset(dataset_type=DatasetType.PREFERENCE, samples=[])
        empty_test = Dataset(dataset_type=DatasetType.PREFERENCE, samples=[])
        return empty_train, empty_test


@step
def push_to_huggingface(
    train_dataset: Dataset,
    test_dataset: Dataset,
    dataset_id: str,
    private: bool = False,
) -> Annotated[str, "push_status"]:
    """
    Push datasets to HuggingFace Hub.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        dataset_id: HuggingFace dataset ID (e.g., "username/dataset-name")
        private: Whether to make the dataset private
        
    Returns:
        Status message
    """
    logger.info(f"Pushing datasets to HuggingFace: {dataset_id}")
    
    try:
        from datasets import Dataset as HFDataset, DatasetDict
        
        # Convert to HuggingFace format
        train_hf = HFDataset.from_dict({
            "samples": [s.model_dump() for s in train_dataset.samples]
        })
        test_hf = HFDataset.from_dict({
            "samples": [s.model_dump() for s in test_dataset.samples]
        })
        
        # Create dataset dict
        dataset_dict = DatasetDict({
            "train": train_hf,
            "test": test_hf,
        })
        
        # Push to hub
        dataset_dict.push_to_hub(
            dataset_id,
            private=private,
        )
        
        logger.success(f"Successfully pushed to HuggingFace: {dataset_id}")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="push_status",
            metadata={
                "dataset_id": dataset_id,
                "train_samples": len(train_dataset.samples),
                "test_samples": len(test_dataset.samples),
                "private": private,
            }
        )
        
        return f"Successfully pushed to {dataset_id}"
        
    except Exception as e:
        logger.error(f"Failed to push to HuggingFace: {e}")
        return f"Failed: {str(e)}"
