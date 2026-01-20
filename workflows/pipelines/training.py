"""Training pipelines for SFT and DPO."""

from pathlib import Path
from zenml import pipeline
from loguru import logger

from tasks.training import (
    train_sft,
    train_dpo,
    load_training_dataset,
    push_model_to_hub,
)


@pipeline
def sft_training_pipeline(
    dataset_path: str,
    model_name: str = "meta-llama/Llama-3.2-1B",
    output_dir: str = "models/sft-model",
    num_train_epochs: int = 3,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> None:
    """
    Supervised Fine-Tuning pipeline.
    
    Args:
        dataset_path: Path to training dataset
        model_name: Base model to fine-tune
        output_dir: Output directory for trained model
        num_train_epochs: Number of training epochs
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_id: HuggingFace repo ID (required if push_to_hub=True)
    """
    logger.info(f"Starting SFT training pipeline for {model_name}")
    
    # Load dataset
    dataset = load_training_dataset(dataset_path=dataset_path)
    
    # Get train/test splits
    if hasattr(dataset, 'keys'):  # DatasetDict
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("test") or dataset.get("validation")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    # Train
    metrics = train_sft(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_name=model_name,
        num_train_epochs=num_train_epochs,
        output_dir=output_dir,
    )
    
    # Push to hub if requested
    if push_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id required when push_to_hub=True")
        
        push_model_to_hub(
            model_path=output_dir,
            repo_id=hub_repo_id,
            commit_message=f"SFT model trained for {num_train_epochs} epochs",
        )
    
    logger.info(f"SFT pipeline completed. Metrics: {metrics}")


@pipeline
def dpo_training_pipeline(
    dataset_path: str,
    model_name: str = "meta-llama/Llama-3.2-1B",
    output_dir: str = "models/dpo-model",
    num_train_epochs: int = 3,
    beta: float = 0.1,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> None:
    """
    Direct Preference Optimization training pipeline.
    
    Args:
        dataset_path: Path to preference dataset
        model_name: Base model to fine-tune
        output_dir: Output directory for trained model
        num_train_epochs: Number of training epochs
        beta: DPO beta parameter
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_id: HuggingFace repo ID (required if push_to_hub=True)
    """
    logger.info(f"Starting DPO training pipeline for {model_name}")
    
    # Load dataset
    dataset = load_training_dataset(dataset_path=dataset_path)
    
    # Get train/test splits
    if hasattr(dataset, 'keys'):  # DatasetDict
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("test") or dataset.get("validation")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    # Train
    metrics = train_dpo(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_name=model_name,
        num_train_epochs=num_train_epochs,
        beta=beta,
        output_dir=output_dir,
    )
    
    # Push to hub if requested
    if push_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id required when push_to_hub=True")
        
        push_model_to_hub(
            model_path=output_dir,
            repo_id=hub_repo_id,
            commit_message=f"DPO model trained for {num_train_epochs} epochs (beta={beta})",
        )
    
    logger.info(f"DPO pipeline completed. Metrics: {metrics}")
