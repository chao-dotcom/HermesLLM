"""
ZenML Pipelines Example Usage

This script demonstrates how to use the various ZenML pipelines
in HermesLLM for different ML workflow scenarios.
"""

from loguru import logger

from hermes.zenml_pipelines import (
    data_collection_pipeline,
    document_processing_pipeline,
    dataset_generation_pipeline,
    model_training_pipeline,
    model_evaluation_pipeline,
    end_to_end_pipeline,
    generate_run_name,
)
from hermes.core.enums import DatasetType


def example_data_collection():
    """Example: Collect data from web sources."""
    logger.info("Running data collection pipeline example")
    
    run_name = generate_run_name("data_collection")
    
    pipeline = data_collection_pipeline.with_options(run_name=run_name)
    pipeline(
        author_full_name="John Doe",
        links=[
            "https://medium.com/@johndoe/article-1",
            "https://github.com/johndoe",
        ],
        platforms=["medium", "github"],
    )


def example_document_processing():
    """Example: Process collected documents."""
    logger.info("Running document processing pipeline example")
    
    run_name = generate_run_name("document_processing")
    
    pipeline = document_processing_pipeline.with_options(run_name=run_name)
    pipeline(
        author_names=["John Doe"],
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        tokens_per_chunk=512,
        collection_name="john_doe_docs",
    )


def example_dataset_generation():
    """Example: Generate instruction dataset."""
    logger.info("Running dataset generation pipeline example")
    
    run_name = generate_run_name("dataset_generation")
    
    pipeline = dataset_generation_pipeline.with_options(run_name=run_name)
    pipeline(
        dataset_type=DatasetType.INSTRUCTION,
        num_samples=100,
        test_split_size=0.2,
        model="gpt-4o-mini",
        push_to_hf=False,
        author_names=["John Doe"],
        mock=True,  # Use mock mode for testing
    )


def example_model_training():
    """Example: Train a model with LoRA."""
    logger.info("Running model training pipeline example")
    
    run_name = generate_run_name("model_training")
    
    pipeline = model_training_pipeline.with_options(run_name=run_name)
    pipeline(
        finetuning_type="sft",
        model_id="meta-llama/Llama-2-7b-hf",
        dataset_id="username/dataset-name",
        num_train_epochs=3,
        learning_rate=3e-4,
        use_lora=True,
        use_4bit=True,
        output_dir="./models/llama2-finetuned",
        push_to_hub=False,
        is_dummy=True,  # Use dummy mode for testing
    )


def example_model_evaluation():
    """Example: Evaluate a trained model."""
    logger.info("Running model evaluation pipeline example")
    
    run_name = generate_run_name("model_evaluation")
    
    pipeline = model_evaluation_pipeline.with_options(run_name=run_name)
    pipeline(
        model_id="username/model-name",
        metrics=["g-eval", "accuracy"],
        benchmarks=["mmlu", "gsm8k"],
        num_eval_samples=100,
        num_benchmark_samples=100,
    )


def example_end_to_end():
    """Example: Run complete end-to-end pipeline."""
    logger.info("Running end-to-end pipeline example")
    
    run_name = generate_run_name("end_to_end")
    
    pipeline = end_to_end_pipeline.with_options(run_name=run_name)
    pipeline(
        # Collection
        author_full_name="John Doe",
        links=["https://medium.com/@johndoe/article-1"],
        platforms=["medium"],
        
        # Processing
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        tokens_per_chunk=512,
        collection_name="john_doe_docs",
        
        # Dataset generation
        dataset_type="instruction",
        num_samples=50,
        test_split_size=0.2,
        dataset_model="gpt-4o-mini",
        push_to_hf=False,
        dataset_id="username/dataset-name",
        
        # Training
        finetuning_type="sft",
        base_model_id="meta-llama/Llama-2-7b-hf",
        num_train_epochs=1,
        use_lora=True,
        training_output_dir="./models/finetuned",
        push_model_to_hub=False,
        
        # Evaluation
        metrics=["g-eval"],
        benchmarks=["mmlu"],
        
        # Control
        skip_training=True,  # Skip expensive training step
        skip_evaluation=True,  # Skip evaluation
        is_dummy=True,  # Use dummy mode
    )


def example_with_config_file():
    """Example: Run pipeline from YAML config."""
    from hermes.zenml_pipelines import load_pipeline_config
    
    logger.info("Running pipeline with config file")
    
    # Load config
    config = load_pipeline_config("configs/data_collection.yaml")
    
    # Run pipeline with config
    run_name = generate_run_name("data_collection_from_config")
    pipeline = data_collection_pipeline.with_options(run_name=run_name)
    pipeline(**config)


def example_orchestrator_configuration():
    """Example: Configure different orchestrators."""
    logger.info("Pipeline orchestrator configuration examples")
    
    # Local orchestrator (default)
    local_pipeline = data_collection_pipeline.with_options(
        run_name="local_run",
    )
    
    # SageMaker orchestrator (requires configuration)
    # sagemaker_pipeline = data_collection_pipeline.with_options(
    #     run_name="sagemaker_run",
    #     orchestrator="sagemaker",
    # )
    
    # Kubernetes orchestrator (requires configuration)
    # k8s_pipeline = data_collection_pipeline.with_options(
    #     run_name="k8s_run",
    #     orchestrator="kubernetes",
    # )
    
    logger.info("See ZenML docs for orchestrator configuration")


if __name__ == "__main__":
    logger.info("ZenML Pipeline Examples")
    logger.info("=" * 50)
    
    # Run individual examples
    # Uncomment the one you want to try:
    
    # example_data_collection()
    # example_document_processing()
    # example_dataset_generation()
    # example_model_training()
    # example_model_evaluation()
    example_end_to_end()
    
    # Advanced examples
    # example_with_config_file()
    # example_orchestrator_configuration()
    
    logger.success("Examples completed!")
