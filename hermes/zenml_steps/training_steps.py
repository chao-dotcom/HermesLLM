"""
Training Steps for ZenML Pipelines

This module contains steps for model training and fine-tuning.
"""

from typing import Dict, Any
from typing_extensions import Annotated

from loguru import logger
from zenml import get_step_context, step

from hermes.training.trainer import ModelTrainer
from hermes.core.enums import FinetuningType


@step
def train_model(
    finetuning_type: str = "sft",
    model_id: str = "meta-llama/Llama-2-7b-hf",
    dataset_id: str = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    use_lora: bool = True,
    use_4bit: bool = True,
    max_seq_length: int = 2048,
    output_dir: str = "./models/finetuned",
    push_to_hub: bool = False,
    hub_model_id: str = None,
    is_dummy: bool = False,
) -> Annotated[Dict[str, Any], "training_results"]:
    """
    Train or fine-tune a language model.
    
    Args:
        finetuning_type: Type of fine-tuning (sft, dpo, orpo)
        model_id: Base model identifier
        dataset_id: HuggingFace dataset ID or local path
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate
        use_lora: Whether to use LoRA
        use_4bit: Whether to use 4-bit quantization
        max_seq_length: Maximum sequence length
        output_dir: Output directory for model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: HuggingFace model ID for pushing
        is_dummy: Whether to use dummy mode (for testing)
        
    Returns:
        Dictionary with training results
    """
    logger.info(f"Starting {finetuning_type} training for {model_id}")
    
    try:
        # Convert string to enum
        ft_type = FinetuningType(finetuning_type.upper())
        
        # Initialize trainer
        trainer = ModelTrainer(
            model_id=model_id,
            finetuning_type=ft_type,
            use_lora=use_lora,
            use_4bit=use_4bit,
        )
        
        if is_dummy:
            logger.warning("Running in dummy mode - no actual training")
            results = {
                "status": "dummy_run",
                "model_id": model_id,
                "finetuning_type": finetuning_type,
                "output_dir": output_dir,
            }
        else:
            # Run training
            results = trainer.train(
                dataset_id=dataset_id,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                output_dir=output_dir,
            )
            
            # Optionally push to hub
            if push_to_hub and hub_model_id:
                trainer.push_to_hub(hub_model_id)
                results["hub_model_id"] = hub_model_id
        
        logger.success(f"Training completed: {results.get('status', 'success')}")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="training_results",
            metadata={
                "finetuning_type": finetuning_type,
                "model_id": model_id,
                "num_epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "use_lora": use_lora,
                "use_4bit": use_4bit,
                "is_dummy": is_dummy,
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }


@step
def deploy_to_sagemaker(
    model_path: str,
    endpoint_name: str,
    instance_type: str = "ml.g5.xlarge",
    instance_count: int = 1,
) -> Annotated[str, "endpoint_name"]:
    """
    Deploy trained model to AWS SageMaker.
    
    Args:
        model_path: Path to trained model
        endpoint_name: SageMaker endpoint name
        instance_type: EC2 instance type
        instance_count: Number of instances
        
    Returns:
        Endpoint name
    """
    logger.info(f"Deploying model to SageMaker endpoint: {endpoint_name}")
    
    try:
        from hermes.deployment.sagemaker import SageMakerDeployment
        
        deployment = SageMakerDeployment()
        
        endpoint = deployment.deploy_model(
            model_path=model_path,
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            instance_count=instance_count,
        )
        
        logger.success(f"Model deployed to endpoint: {endpoint_name}")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="endpoint_name",
            metadata={
                "endpoint_name": endpoint_name,
                "instance_type": instance_type,
                "instance_count": instance_count,
            }
        )
        
        return endpoint_name
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return f"Failed: {str(e)}"
