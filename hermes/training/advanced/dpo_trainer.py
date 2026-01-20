"""Direct Preference Optimization (DPO) trainer."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger

from hermes.training.advanced.config import TrainingConfig
from hermes.training.advanced.model_loader import ModelLoader

# Import DPO training dependencies
try:
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()  # Apply patches
    PATCH_APPLIED = True
except ImportError:
    PATCH_APPLIED = False
    logger.warning("Unsloth DPO patches not available")

try:
    from transformers import TrainingArguments
    from trl import DPOTrainer as HFDPOTrainer, DPOConfig as HFDPOConfig
    from datasets import load_dataset, Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    logger.warning("TRL or datasets not installed")


class DPOTrainer:
    """Direct Preference Optimization trainer with LoRA."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize DPO trainer.
        
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
        logger.info("Loading model and tokenizer for DPO...")
        
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
        Prepare preference dataset for DPO training.
        
        Dataset should have columns: instruction/prompt, chosen, rejected
        
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
        
        # Format for DPO (ensure proper column names)
        def format_sample(sample):
            """Format sample for DPO training."""
            # Map various field names to standard format
            prompt = sample.get("instruction") or sample.get("prompt")
            chosen = sample.get("chosen") or sample.get("preferred_answer")
            rejected = sample.get("rejected") or sample.get("rejected_answer")
            
            if not all([prompt, chosen, rejected]):
                raise ValueError(f"Missing required fields. Got: {sample.keys()}")
            
            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        
        # Apply formatting
        dataset = dataset.map(format_sample)
        
        # Ensure we have the right columns
        required_cols = {"prompt", "chosen", "rejected"}
        if not required_cols.issubset(set(dataset.column_names)):
            raise ValueError(f"Dataset missing required columns. Has: {dataset.column_names}, needs: {required_cols}")
        
        logger.info("Dataset prepared for DPO training")
        
        return dataset
    
    def train(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """
        Run DPO training.
        
        Args:
            train_dataset: Training dataset with prompt, chosen, rejected columns
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Training metrics
        """
        logger.info("Starting Direct Preference Optimization training...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Prepare datasets
        if train_dataset is None:
            train_dataset = self.prepare_dataset()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure DPO-specific arguments
        dpo_config = HFDPOConfig(
            **self.config.to_training_arguments(),
            beta=self.config.dpo.beta if self.config.dpo else 0.1,
            label_smoothing=self.config.dpo.label_smoothing if self.config.dpo else 0.0,
            loss_type=self.config.dpo.loss_type if self.config.dpo else "sigmoid",
            report_to=["tensorboard"],
            logging_dir=str(self.config.output_dir / "logs"),
            remove_unused_columns=False,
        )
        
        # Initialize DPO trainer
        self.trainer = HFDPOTrainer(
            model=self.model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
            max_prompt_length=self.config.max_seq_length // 2,
        )
        
        # Train
        logger.info(f"Training for {self.config.num_train_epochs} epochs with DPO...")
        logger.info(f"DPO beta: {dpo_config.beta}")
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
        
        logger.info(f"DPO training complete! Metrics: {metrics}")
        
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
