"""Dataset models for training."""

from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets
    
    DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("HuggingFace datasets not installed")
    DATASETS_AVAILABLE = False

from .base import VectorDocument
from .enums import DataCategory


class DatasetType(str, Enum):
    """Types of training datasets."""
    
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"
    COMPLETION = "completion"


class InstructSample(BaseModel):
    """Single instruction-following sample."""
    
    instruction: str = Field(description="Input instruction/question")
    output: str = Field(description="Expected output/answer")
    context: str | None = Field(None, description="Optional context")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferenceSample(BaseModel):
    """Single preference learning sample (DPO/RLHF)."""
    
    instruction: str = Field(description="Input instruction")
    chosen: str = Field(description="Preferred response")
    rejected: str = Field(description="Rejected response")
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstructDataset(VectorDocument):
    """Collection of instruction-following samples."""
    
    samples: list[InstructSample] = Field(default_factory=list)
    category: DataCategory
    source: str | None = None
    
    class Config:
        category = DataCategory.INSTRUCT_DATASET
    
    @property
    def num_samples(self) -> int:
        """Get number of samples in dataset."""
        return len(self.samples)
    
    def to_huggingface(self) -> "Dataset":
        """Convert to HuggingFace Dataset."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library not installed")
        
        data = {
            "instruction": [s.instruction for s in self.samples],
            "output": [s.output for s in self.samples],
        }
        
        # Add context if available
        if any(s.context for s in self.samples):
            data["context"] = [s.context or "" for s in self.samples]
        
        return Dataset.from_dict(data)
    
    def add_sample(self, instruction: str, output: str, context: str | None = None) -> None:
        """Add a new sample to the dataset."""
        self.samples.append(
            InstructSample(
                instruction=instruction,
                output=output,
                context=context
            )
        )


class PreferenceDataset(VectorDocument):
    """Collection of preference learning samples."""
    
    samples: list[PreferenceSample] = Field(default_factory=list)
    category: DataCategory
    source: str | None = None
    
    class Config:
        category = DataCategory.PREFERENCE_DATASET
    
    @property
    def num_samples(self) -> int:
        """Get number of samples in dataset."""
        return len(self.samples)
    
    def to_huggingface(self) -> "Dataset":
        """Convert to HuggingFace Dataset."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library not installed")
        
        data = {
            "instruction": [s.instruction for s in self.samples],
            "chosen": [s.chosen for s in self.samples],
            "rejected": [s.rejected for s in self.samples],
        }
        
        return Dataset.from_dict(data)
    
    def add_sample(self, instruction: str, chosen: str, rejected: str) -> None:
        """Add a new sample to the dataset."""
        self.samples.append(
            PreferenceSample(
                instruction=instruction,
                chosen=chosen,
                rejected=rejected
            )
        )


class TrainTestSplit(BaseModel):
    """Train/test split of datasets."""
    
    train: dict[str, InstructDataset | PreferenceDataset]
    test: dict[str, InstructDataset | PreferenceDataset]
    test_split_size: float = 0.1
    
    def to_huggingface(self, flatten: bool = False) -> "DatasetDict":
        """
        Convert to HuggingFace DatasetDict.
        
        Args:
            flatten: If True, concatenate all categories into single train/test sets
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library not installed")
        
        train_datasets = {
            category: dataset.to_huggingface()
            for category, dataset in self.train.items()
        }
        test_datasets = {
            category: dataset.to_huggingface()
            for category, dataset in self.test.items()
        }
        
        if flatten:
            train_data = concatenate_datasets(list(train_datasets.values()))
            test_data = concatenate_datasets(list(test_datasets.values()))
        else:
            train_data = Dataset.from_dict(train_datasets)
            test_data = Dataset.from_dict(test_datasets)
        
        return DatasetDict({"train": train_data, "test": test_data})
    
    @property
    def train_size(self) -> int:
        """Total training samples."""
        return sum(d.num_samples for d in self.train.values())
    
    @property
    def test_size(self) -> int:
        """Total test samples."""
        return sum(d.num_samples for d in self.test.values())
