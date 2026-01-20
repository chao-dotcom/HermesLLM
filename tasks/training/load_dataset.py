"""Load training dataset for fine-tuning."""

from typing import Optional
from zenml import step
from loguru import logger

try:
    from datasets import load_dataset, Dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not installed")


@step
def load_training_dataset(
    dataset_path: str,
    split: Optional[str] = None,
    train_split: str = "train",
    test_split: str = "test",
) -> Dataset | DatasetDict:
    """
    Load training dataset from HuggingFace Hub or local path.
    
    Args:
        dataset_path: Path to dataset (HF Hub or local)
        split: Specific split to load (None loads all)
        train_split: Name of training split
        test_split: Name of test split
        
    Returns:
        Dataset or DatasetDict
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. Install: pip install datasets")
    
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Load dataset
    if split:
        dataset = load_dataset(dataset_path, split=split)
        logger.info(f"Loaded {len(dataset)} samples from split '{split}'")
    else:
        dataset = load_dataset(dataset_path)
        if isinstance(dataset, DatasetDict):
            total = sum(len(ds) for ds in dataset.values())
            logger.info(f"Loaded {total} total samples across {len(dataset)} splits")
        else:
            logger.info(f"Loaded {len(dataset)} samples")
    
    return dataset
