"""
End-to-End Pipeline

Orchestrates the complete ML workflow from data collection to model deployment.
"""

from typing import List, Optional

from zenml import pipeline

from hermes.zenml_pipelines.collection_pipeline import data_collection_pipeline
from hermes.zenml_pipelines.processing_pipeline import document_processing_pipeline
from hermes.zenml_pipelines.dataset_pipeline import dataset_generation_pipeline
from hermes.zenml_pipelines.training_pipeline import model_training_pipeline
from hermes.zenml_pipelines.evaluation_pipeline import model_evaluation_pipeline
from hermes.core.enums import DatasetType


@pipeline(name="end_to_end_pipeline")
def end_to_end_pipeline(
    # Collection parameters
    author_full_name: str,
    links: List[str] = None,
    platforms: List[str] = None,
    
    # Processing parameters
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk: int = 512,
    collection_name: str = "documents",
    
    # Dataset generation parameters
    dataset_type: str = "instruction",
    num_samples: int = 100,
    test_split_size: float = 0.2,
    dataset_model: str = "gpt-4o-mini",
    push_to_hf: bool = False,
    dataset_id: str = None,
    
    # Training parameters
    finetuning_type: str = "sft",
    base_model_id: str = "meta-llama/Llama-2-7b-hf",
    num_train_epochs: int = 3,
    use_lora: bool = True,
    training_output_dir: str = "./models/finetuned",
    push_model_to_hub: bool = False,
    hub_model_id: str = None,
    
    # Evaluation parameters
    eval_dataset_id: str = None,
    metrics: List[str] = None,
    benchmarks: List[str] = None,
    
    # Deployment parameters
    deploy_to_sagemaker: bool = False,
    endpoint_name: str = None,
    
    # Control flags
    skip_collection: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    is_dummy: bool = False,
) -> str:
    """
    Complete end-to-end ML pipeline.
    
    This orchestrates the full workflow:
    1. Data collection from various sources
    2. Document processing (cleaning, chunking, embedding)
    3. Dataset generation for training
    4. Model training/fine-tuning
    5. Model evaluation
    6. Optional deployment to SageMaker
    
    Args:
        author_full_name: Author name for data collection
        links: URLs to collect data from
        platforms: Platforms to collect from
        model_name: Embedding model name
        tokens_per_chunk: Tokens per chunk
        collection_name: Vector DB collection name
        dataset_type: Type of dataset (instruction/preference)
        num_samples: Number of samples to generate
        test_split_size: Test split ratio
        dataset_model: Model for dataset generation
        push_to_hf: Whether to push dataset to HuggingFace
        dataset_id: HuggingFace dataset ID
        finetuning_type: Type of fine-tuning
        base_model_id: Base model for training
        num_train_epochs: Number of training epochs
        use_lora: Whether to use LoRA
        training_output_dir: Output directory for trained model
        push_model_to_hub: Whether to push model to HuggingFace
        hub_model_id: HuggingFace model ID
        eval_dataset_id: Evaluation dataset ID
        metrics: Evaluation metrics
        benchmarks: Benchmark names
        deploy_to_sagemaker: Whether to deploy to SageMaker
        endpoint_name: SageMaker endpoint name
        skip_collection: Skip collection step
        skip_training: Skip training step
        skip_evaluation: Skip evaluation step
        is_dummy: Dummy mode for testing
        
    Returns:
        Final step invocation ID
    """
    # Step 1: Data Collection (optional)
    if not skip_collection and links:
        collection_invocation = data_collection_pipeline(
            author_full_name=author_full_name,
            links=links,
            platforms=platforms,
        )
    else:
        collection_invocation = None
    
    # Step 2: Document Processing
    processing_invocations = document_processing_pipeline(
        author_names=[author_full_name],
        model_name=model_name,
        tokens_per_chunk=tokens_per_chunk,
        collection_name=collection_name,
        wait_for=collection_invocation if collection_invocation else None,
    )
    
    # Step 3: Dataset Generation
    dataset_type_enum = DatasetType(dataset_type.upper())
    dataset_invocation = dataset_generation_pipeline(
        dataset_type=dataset_type_enum,
        num_samples=num_samples,
        test_split_size=test_split_size,
        model=dataset_model,
        push_to_hf=push_to_hf,
        dataset_id=dataset_id,
        author_names=[author_full_name],
        mock=is_dummy,
        wait_for=processing_invocations[-1] if processing_invocations else None,
    )
    
    # Step 4: Model Training (optional)
    if not skip_training:
        training_invocation = model_training_pipeline(
            finetuning_type=finetuning_type,
            model_id=base_model_id,
            dataset_id=dataset_id,
            num_train_epochs=num_train_epochs,
            use_lora=use_lora,
            output_dir=training_output_dir,
            push_to_hub=push_model_to_hub,
            hub_model_id=hub_model_id,
            deploy_to_sagemaker_flag=deploy_to_sagemaker,
            endpoint_name=endpoint_name,
            is_dummy=is_dummy,
        )
    else:
        training_invocation = None
    
    # Step 5: Model Evaluation (optional)
    if not skip_evaluation and training_invocation:
        # Use the trained model ID
        eval_model_id = hub_model_id if push_model_to_hub else base_model_id
        evaluation_invocation = model_evaluation_pipeline(
            model_id=eval_model_id,
            eval_dataset_id=eval_dataset_id,
            metrics=metrics,
            benchmarks=benchmarks,
        )
        return evaluation_invocation
    
    # Return the last successful invocation
    if training_invocation:
        return training_invocation
    elif dataset_invocation:
        return dataset_invocation
    elif processing_invocations:
        return processing_invocations[-1]
    else:
        return collection_invocation if collection_invocation else "no_ops"
