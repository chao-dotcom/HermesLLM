"""Model training orchestration."""

from typing import Dict, Any, Optional
from pathlib import Path

from loguru import logger

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not installed. Install with: pip install transformers datasets")


class LLMTrainer:
    """Trainer for fine-tuning language models."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: str | Path = "models/fine-tuned"
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            model_name: Base model name
            output_dir: Output directory for trained model
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required. Install: pip install transformers datasets torch")
        
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing trainer for {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def prepare_dataset(
        self,
        data: list[Dict[str, Any]],
        text_column: str = "text"
    ) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            data: List of data dictionaries
            text_column: Column containing text
            
        Returns:
            Prepared dataset
        """
        logger.info(f"Preparing dataset with {len(data)} examples")
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(data)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info("Dataset prepared and tokenized")
        return tokenized
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        save_steps: int = 500
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_steps: Save checkpoint every N steps
            
        Returns:
            Training metrics
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=save_steps,
            save_total_limit=2,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            report_to=["tensorboard"],
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Training complete. Model saved to {self.output_dir}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        }
