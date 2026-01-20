"""Supervised Fine-Tuning (SFT) trainer with LoRA support."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger

from hermes.training.advanced.config import TrainingConfig
from hermes.training.advanced.model_loader import ModelLoader

# Try importing training libraries
try:
    from transformers import TrainingArguments
    from trl import SFTTrainer as HFSFTTrainer
    from datasets import load_dataset, Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    logger.warning("TRL or datasets not installed. Install with: pip install trl datasets")


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


class SFTTrainer:
    """Supervised Fine-Tuning trainer with LoRA."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize SFT trainer.
        
        Args:
            config: Training configuration
        """
        if not TRAINING_AVAILABLE:
            raise ImportError("TRL and datasets required. Install: pip install trl datasets")
        
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info("Loading model and tokenizer...")
        
        loader = ModelLoader(self.config)
        self.model, self.tokenizer = loader.load_model_and_tokenizer()
        
        # Ensure EOS token is set
        if not hasattr(self.tokenizer, 'eos_token') or self.tokenizer.eos_token is None:
            logger.warning("EOS token not found, using default")
            self.tokenizer.eos_token = "</s>"
        
        logger.info(f"EOS token: {self.tokenizer.eos_token}")
    
    def prepare_dataset(
        self, 
        dataset_path: Optional[str] = None,
        dataset: Optional[Dataset] = None,
    ) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            dataset_path: Path to HuggingFace dataset
            dataset: Pre-loaded dataset object
            
        Returns:
            Prepared dataset
        """
        if dataset is not None:
            logger.info(f"Using provided dataset with {len(dataset)} samples")
            return dataset
        
        if dataset_path is None:
            dataset_path = self.config.dataset_path
        
        if dataset_path is None:
            raise ValueError("Either dataset_path or dataset must be provided")
        
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Load dataset
        if os.path.exists(dataset_path):
            # Local file
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        else:
            # HuggingFace Hub
            dataset = load_dataset(dataset_path, split="train")
        
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Format dataset for SFT
        def format_sample(sample):
            """Format sample for training."""
            if "instruction" in sample and "output" in sample:
                # Instruction format
                text = ALPACA_TEMPLATE.format(
                    sample["instruction"],
                    sample["output"]
                ) + self.tokenizer.eos_token
            elif "text" in sample:
                # Raw text format
                text = sample["text"] + self.tokenizer.eos_token
            else:
                raise ValueError(f"Unknown dataset format: {sample.keys()}")
            
            return {"text": text}
        
        # Apply formatting
        dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
        
        logger.info("Dataset prepared for SFT training")
        
        return dataset
    
    def train(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """
        Run supervised fine-tuning.
        
        Args:
            train_dataset: Training dataset (optional if config has dataset_path)
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Training metrics
        """
        logger.info("Starting supervised fine-tuning...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Prepare datasets
        if train_dataset is None:
            train_dataset = self.prepare_dataset()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure training arguments
        training_args = TrainingArguments(
            **self.config.to_training_arguments(),
            report_to=["tensorboard"],
            logging_dir=str(self.config.output_dir / "logs"),
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        self.trainer = HFSFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=False,  # Disable packing for simpler debugging
        )
        
        # Train
        logger.info(f"Training for {self.config.num_train_epochs} epochs...")
        train_result = self.trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.config.output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Extract metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }
        
        logger.info(f"Training complete! Metrics: {metrics}")
        
        return metrics
    
    def save_model(self, output_path: Path | str) -> None:
        """
        Save trained model.
        
        Args:
            output_path: Path to save model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.trainer is not None:
            self.trainer.save_model(str(output_path))
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Model saved to {output_path}")
        else:
            logger.warning("No trainer available, nothing to save")
