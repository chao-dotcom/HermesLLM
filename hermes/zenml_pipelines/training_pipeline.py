"""
Training Pipeline

ZenML pipeline for model training and fine-tuning.
"""

from typing import Optional

from zenml import pipeline

from hermes.zenml_steps.training_steps import train_model, deploy_to_sagemaker


@pipeline(name="model_training_pipeline")
def model_training_pipeline(
    finetuning_type: str = "sft",
    model_id: str = "meta-llama/Llama-2-7b-hf",
    dataset_id: str = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    use_lora: bool = True,
    use_4bit: bool = True,
    output_dir: str = "./models/finetuned",
    push_to_hub: bool = False,
    hub_model_id: str = None,
    deploy_to_sagemaker_flag: bool = False,
    endpoint_name: str = None,
    instance_type: str = "ml.g5.xlarge",
    is_dummy: bool = False,
) -> Optional[str]:
    """
    Pipeline for training and optionally deploying a language model.
    
    This pipeline:
    1. Trains/fine-tunes a model using specified configuration
    2. Optionally pushes to HuggingFace Hub
    3. Optionally deploys to AWS SageMaker
    
    Args:
        finetuning_type: Type of fine-tuning (sft, dpo, orpo)
        model_id: Base model identifier
        dataset_id: Training dataset ID
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate
        use_lora: Whether to use LoRA
        use_4bit: Whether to use 4-bit quantization
        output_dir: Output directory
        push_to_hub: Whether to push to HuggingFace
        hub_model_id: HuggingFace model ID
        deploy_to_sagemaker_flag: Whether to deploy to SageMaker
        endpoint_name: SageMaker endpoint name
        instance_type: SageMaker instance type
        is_dummy: Dummy mode for testing
        
    Returns:
        Optional endpoint name if deployed
    """
    # Step 1: Train model
    training_results = train_model(
        finetuning_type=finetuning_type,
        model_id=model_id,
        dataset_id=dataset_id,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        use_lora=use_lora,
        use_4bit=use_4bit,
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        is_dummy=is_dummy,
    )
    
    # Step 2: Optionally deploy to SageMaker
    if deploy_to_sagemaker_flag and endpoint_name:
        endpoint = deploy_to_sagemaker(
            model_path=output_dir,
            endpoint_name=endpoint_name,
            instance_type=instance_type,
        )
        return endpoint.invocation_id
    
    return None
