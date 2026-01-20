"""Supervised Fine-Tuning step."""

from pathlib import Path
from typing import Dict, Any, Optional
from zenml import step
from loguru import logger

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from hermes.training.advanced import TrainingConfig, SFTTrainer


@step
def train_sft(
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[TrainingConfig] = None,
    # Model config
    model_name: str = "meta-llama/Llama-3.2-1B",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    # LoRA config
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    # Training config
    learning_rate: float = 3e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    # Output
    output_dir: str = "models/sft-model",
    dataset_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run supervised fine-tuning with LoRA.
    
    Args:
        train_dataset: Training dataset (optional if dataset_path provided)
        eval_dataset: Evaluation dataset (optional)
        config: Complete training configuration (overrides individual params)
        model_name: Base model to fine-tune
        max_seq_length: Maximum sequence length
        load_in_4bit: Use 4-bit quantization
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: Learning rate
        num_train_epochs: Number of epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        output_dir: Output directory
        dataset_path: Path to dataset (if not providing dataset directly)
        
    Returns:
        Training metrics
    """
    logger.info("Starting SFT training step...")
    
    # Create config if not provided
    if config is None:
        config = TrainingConfig(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            output_dir=Path(output_dir),
            dataset_path=dataset_path,
        )
        
        # Update LoRA config
        config.lora.rank = lora_rank
        config.lora.alpha = lora_alpha
        config.lora.dropout = lora_dropout
    
    # Initialize trainer
    trainer = SFTTrainer(config)
    
    # Train
    metrics = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    logger.info(f"SFT training completed with metrics: {metrics}")
    
    return metrics
