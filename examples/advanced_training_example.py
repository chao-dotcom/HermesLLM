"""
Advanced Training Example
==========================

This example demonstrates how to use the advanced training features including:
- Supervised Fine-Tuning (SFT) with LoRA
- Direct Preference Optimization (DPO)
- 4-bit quantization with Unsloth
- ZenML pipeline orchestration
"""

from pathlib import Path
from loguru import logger

from hermes.training.advanced import TrainingConfig, LoRAConfig, DPOConfig, SFTTrainer, DPOTrainer
from workflows.pipelines.training import sft_training_pipeline, dpo_training_pipeline


def example_sft_training():
    """Example: Supervised Fine-Tuning with LoRA."""
    
    logger.info("=== SFT Training Example ===")
    
    # Configure training
    config = TrainingConfig(
        # Model settings
        model_name="meta-llama/Llama-3.2-1B",  # Base model
        max_seq_length=2048,
        load_in_4bit=True,  # Use 4-bit quantization
        
        # LoRA settings
        lora=LoRAConfig(
            rank=32,  # LoRA rank
            alpha=32,  # LoRA alpha (typically same as rank)
            dropout=0.0,  # LoRA dropout
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which layers to adapt
        ),
        
        # Training hyperparameters
        learning_rate=3e-4,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
        warmup_steps=100,
        
        # Optimization
        optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Chat template (for instruction datasets)
        chat_template="chatml",  # Options: alpaca, chatml, llama2, llama3
        
        # Output
        output_dir=Path("models/my-sft-model"),
        
        # Dataset (can also pass directly to train())
        dataset_path="username/my-instruction-dataset",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(config)
    
    # Train (dataset loaded from config.dataset_path)
    metrics = trainer.train()
    
    logger.info(f"Training completed! Metrics: {metrics}")
    logger.info(f"Model saved to: {config.output_dir}")


def example_dpo_training():
    """Example: Direct Preference Optimization."""
    
    logger.info("=== DPO Training Example ===")
    
    # Configure training
    config = TrainingConfig(
        # Model settings
        model_name="meta-llama/Llama-3.2-1B",
        max_seq_length=2048,
        load_in_4bit=True,
        
        # LoRA settings
        lora=LoRAConfig(
            rank=32,
            alpha=32,
            dropout=0.0,
        ),
        
        # DPO-specific settings
        dpo=DPOConfig(
            beta=0.1,  # DPO temperature parameter (higher = stronger preference)
            loss_type="sigmoid",  # Options: sigmoid, hinge, ipo
        ),
        
        # Training hyperparameters
        learning_rate=5e-5,  # Often lower than SFT
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        
        # Output
        output_dir=Path("models/my-dpo-model"),
        dataset_path="username/my-preference-dataset",
    )
    
    # Initialize trainer
    trainer = DPOTrainer(config)
    
    # Train (dataset must have prompt, chosen, rejected columns)
    metrics = trainer.train()
    
    logger.info(f"DPO training completed! Metrics: {metrics}")


def example_with_custom_dataset():
    """Example: Training with custom dataset object."""
    
    from datasets import Dataset
    
    logger.info("=== Custom Dataset Example ===")
    
    # Create sample dataset
    train_data = {
        "instruction": ["What is Python?", "Explain machine learning"],
        "input": ["", ""],
        "output": [
            "Python is a high-level programming language...",
            "Machine learning is a subset of AI...",
        ],
    }
    train_dataset = Dataset.from_dict(train_data)
    
    # Configure training
    config = TrainingConfig(
        model_name="meta-llama/Llama-3.2-1B",
        output_dir=Path("models/custom-model"),
        num_train_epochs=1,
    )
    
    # Train with custom dataset
    trainer = SFTTrainer(config)
    metrics = trainer.train(train_dataset=train_dataset)
    
    logger.info(f"Training with custom dataset completed: {metrics}")


def example_zenml_pipeline():
    """Example: Run training as ZenML pipeline."""
    
    logger.info("=== ZenML Pipeline Example ===")
    
    # Run SFT pipeline
    sft_training_pipeline(
        dataset_path="username/my-instruction-dataset",
        model_name="meta-llama/Llama-3.2-1B",
        output_dir="models/sft-llama",
        num_train_epochs=3,
        push_to_hub=True,  # Push to HuggingFace Hub
        hub_repo_id="username/my-finetuned-llama",
    )
    
    logger.info("SFT pipeline completed!")
    
    # Run DPO pipeline
    dpo_training_pipeline(
        dataset_path="username/my-preference-dataset",
        model_name="username/my-finetuned-llama",  # Use SFT model as base
        output_dir="models/dpo-llama",
        num_train_epochs=3,
        beta=0.1,
        push_to_hub=True,
        hub_repo_id="username/my-dpo-llama",
    )
    
    logger.info("DPO pipeline completed!")


def example_resume_training():
    """Example: Resume training from checkpoint."""
    
    logger.info("=== Resume Training Example ===")
    
    config = TrainingConfig(
        model_name="meta-llama/Llama-3.2-1B",
        output_dir=Path("models/my-model"),
        resume_from_checkpoint=True,  # Will look for checkpoint in output_dir
        num_train_epochs=5,
    )
    
    trainer = SFTTrainer(config)
    metrics = trainer.train()
    
    logger.info(f"Resumed training completed: {metrics}")


def example_without_unsloth():
    """Example: Train without Unsloth (fallback to transformers)."""
    
    logger.info("=== Training without Unsloth ===")
    
    # Same config works with or without Unsloth
    # The trainer will automatically use transformers if Unsloth is not available
    config = TrainingConfig(
        model_name="meta-llama/Llama-3.2-1B",
        output_dir=Path("models/transformers-model"),
        load_in_4bit=True,  # Still supported via transformers
    )
    
    trainer = SFTTrainer(config)
    metrics = trainer.train()
    
    logger.info(f"Training with transformers completed: {metrics}")


if __name__ == "__main__":
    # Choose which example to run
    
    # Basic SFT training
    # example_sft_training()
    
    # DPO training
    # example_dpo_training()
    
    # Custom dataset
    # example_with_custom_dataset()
    
    # ZenML pipeline
    # example_zenml_pipeline()
    
    # Resume from checkpoint
    # example_resume_training()
    
    # Without Unsloth
    # example_without_unsloth()
    
    logger.info("Uncomment one of the examples above to run it!")
