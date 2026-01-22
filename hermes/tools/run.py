"""
HermesLLM Pipeline Runner

Comprehensive CLI tool for running all HermesLLM pipelines.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from hermes.zenml_pipelines import (
    data_collection_pipeline,
    document_processing_pipeline,
    dataset_generation_pipeline,
    model_training_pipeline,
    model_evaluation_pipeline,
    end_to_end_pipeline,
    load_pipeline_config,
    generate_run_name,
)
from hermes.core.enums import DatasetType


@click.group()
@click.version_option(version="1.0.0", prog_name="HermesLLM")
def cli():
    """
    HermesLLM CLI v1.0.0
    
    Comprehensive command-line interface for running ML pipelines,
    from data collection to model deployment.
    
    Examples:
    
        \b
        # Run complete end-to-end pipeline
        hermes run end-to-end --author "John Doe" --links url1.com url2.com
        
        \b
        # Collect data from web sources
        hermes run collect --author "John Doe" --links url1.com url2.com
        
        \b
        # Process documents
        hermes run process --authors "John Doe" "Jane Smith"
        
        \b
        # Generate instruction dataset
        hermes run generate-dataset --type instruction --samples 100
        
        \b
        # Train a model
        hermes run train --model meta-llama/Llama-2-7b-hf --dataset my-dataset
        
        \b
        # Evaluate a model
        hermes run evaluate --model my-model --benchmarks mmlu gsm8k
    """
    pass


@cli.group()
def run():
    """Run ML pipelines."""
    pass


@run.command(name="end-to-end")
@click.option("--author", required=True, help="Author full name")
@click.option("--links", multiple=True, help="URLs to collect data from")
@click.option("--platforms", multiple=True, default=["medium", "github"], help="Platforms to collect from")
@click.option("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
@click.option("--tokens-per-chunk", default=512, type=int, help="Tokens per chunk")
@click.option("--dataset-type", default="instruction", type=click.Choice(["instruction", "preference"]), help="Dataset type")
@click.option("--samples", default=100, type=int, help="Number of dataset samples")
@click.option("--test-split", default=0.2, type=float, help="Test split ratio")
@click.option("--dataset-model", default="gpt-4o-mini", help="Model for dataset generation")
@click.option("--finetuning-type", default="sft", type=click.Choice(["sft", "dpo", "orpo"]), help="Fine-tuning type")
@click.option("--base-model", default="meta-llama/Llama-2-7b-hf", help="Base model for training")
@click.option("--epochs", default=3, type=int, help="Training epochs")
@click.option("--use-lora/--no-lora", default=True, help="Use LoRA")
@click.option("--skip-collection", is_flag=True, help="Skip data collection")
@click.option("--skip-training", is_flag=True, help="Skip model training")
@click.option("--skip-evaluation", is_flag=True, help="Skip model evaluation")
@click.option("--dummy", is_flag=True, help="Run in dummy mode (testing)")
@click.option("--config", type=Path, help="Load from YAML config file")
@click.option("--no-cache", is_flag=True, help="Disable pipeline caching")
def run_end_to_end(
    author: str,
    links: tuple,
    platforms: tuple,
    model: str,
    tokens_per_chunk: int,
    dataset_type: str,
    samples: int,
    test_split: float,
    dataset_model: str,
    finetuning_type: str,
    base_model: str,
    epochs: int,
    use_lora: bool,
    skip_collection: bool,
    skip_training: bool,
    skip_evaluation: bool,
    dummy: bool,
    config: Optional[Path],
    no_cache: bool,
):
    """Run complete end-to-end ML pipeline."""
    logger.info("Starting end-to-end pipeline")
    
    # Load config if provided
    if config:
        logger.info(f"Loading config from {config}")
        config_data = load_pipeline_config(str(config))
    else:
        config_data = {}
    
    # Prepare pipeline arguments
    pipeline_args = {
        "author_full_name": author,
        "links": list(links) if links else config_data.get("links", []),
        "platforms": list(platforms) if platforms else config_data.get("platforms", ["medium", "github"]),
        "model_name": model,
        "tokens_per_chunk": tokens_per_chunk,
        "dataset_type": dataset_type,
        "num_samples": samples,
        "test_split_size": test_split,
        "dataset_model": dataset_model,
        "finetuning_type": finetuning_type,
        "base_model_id": base_model,
        "num_train_epochs": epochs,
        "use_lora": use_lora,
        "skip_collection": skip_collection,
        "skip_training": skip_training,
        "skip_evaluation": skip_evaluation,
        "is_dummy": dummy,
    }
    
    # Merge with config
    pipeline_args.update(config_data)
    
    # Run pipeline
    run_name = generate_run_name("end_to_end")
    pipeline = end_to_end_pipeline.with_options(
        run_name=run_name,
        enable_cache=not no_cache,
    )
    
    logger.info(f"Running pipeline: {run_name}")
    result = pipeline(**pipeline_args)
    logger.success(f"Pipeline completed: {result}")


@run.command(name="collect")
@click.option("--author", required=True, help="Author full name")
@click.option("--links", multiple=True, required=True, help="URLs to collect data from")
@click.option("--platforms", multiple=True, default=["medium", "github"], help="Platforms to collect from")
@click.option("--config", type=Path, help="Load from YAML config file")
@click.option("--no-cache", is_flag=True, help="Disable pipeline caching")
def run_collect(author: str, links: tuple, platforms: tuple, config: Optional[Path], no_cache: bool):
    """Run data collection pipeline."""
    logger.info("Starting data collection pipeline")
    
    if config:
        config_data = load_pipeline_config(str(config))
        author = config_data.get("author_full_name", author)
        links = config_data.get("links", list(links))
        platforms = config_data.get("platforms", list(platforms))
    
    run_name = generate_run_name("data_collection")
    pipeline = data_collection_pipeline.with_options(
        run_name=run_name,
        enable_cache=not no_cache,
    )
    
    result = pipeline(
        author_full_name=author,
        links=list(links),
        platforms=list(platforms),
    )
    logger.success(f"Collection completed: {result}")


@run.command(name="process")
@click.option("--authors", multiple=True, required=True, help="Author names to process")
@click.option("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
@click.option("--tokens-per-chunk", default=512, type=int, help="Tokens per chunk")
@click.option("--collection", default="documents", help="Vector DB collection name")
@click.option("--limit", type=int, help="Limit number of documents")
@click.option("--no-cache", is_flag=True, help="Disable pipeline caching")
def run_process(authors: tuple, model: str, tokens_per_chunk: int, collection: str, limit: Optional[int], no_cache: bool):
    """Run document processing pipeline."""
    logger.info("Starting document processing pipeline")
    
    run_name = generate_run_name("document_processing")
    pipeline = document_processing_pipeline.with_options(
        run_name=run_name,
        enable_cache=not no_cache,
    )
    
    result = pipeline(
        author_names=list(authors),
        model_name=model,
        tokens_per_chunk=tokens_per_chunk,
        collection_name=collection,
        limit=limit,
    )
    logger.success(f"Processing completed: {result}")


@run.command(name="generate-dataset")
@click.option("--type", "dataset_type", required=True, type=click.Choice(["instruction", "preference"]), help="Dataset type")
@click.option("--samples", default=100, type=int, help="Number of samples to generate")
@click.option("--test-split", default=0.2, type=float, help="Test split ratio")
@click.option("--model", default="gpt-4o-mini", help="Model for generation")
@click.option("--authors", multiple=True, help="Author names to generate from")
@click.option("--push-to-hf", is_flag=True, help="Push to HuggingFace Hub")
@click.option("--dataset-id", help="HuggingFace dataset ID")
@click.option("--mock", is_flag=True, help="Use mock mode")
@click.option("--no-cache", is_flag=True, help="Disable pipeline caching")
def run_generate_dataset(
    dataset_type: str,
    samples: int,
    test_split: float,
    model: str,
    authors: tuple,
    push_to_hf: bool,
    dataset_id: Optional[str],
    mock: bool,
    no_cache: bool,
):
    """Run dataset generation pipeline."""
    logger.info(f"Starting {dataset_type} dataset generation pipeline")
    
    dt = DatasetType(dataset_type.upper())
    
    run_name = generate_run_name(f"dataset_generation_{dataset_type}")
    pipeline = dataset_generation_pipeline.with_options(
        run_name=run_name,
        enable_cache=not no_cache,
    )
    
    result = pipeline(
        dataset_type=dt,
        num_samples=samples,
        test_split_size=test_split,
        model=model,
        author_names=list(authors) if authors else None,
        push_to_hf=push_to_hf,
        dataset_id=dataset_id,
        mock=mock,
    )
    logger.success(f"Dataset generation completed: {result}")


@run.command(name="train")
@click.option("--model", required=True, help="Base model ID")
@click.option("--dataset", required=True, help="Training dataset ID")
@click.option("--finetuning-type", default="sft", type=click.Choice(["sft", "dpo", "orpo"]), help="Fine-tuning type")
@click.option("--epochs", default=3, type=int, help="Number of epochs")
@click.option("--batch-size", default=2, type=int, help="Batch size per device")
@click.option("--learning-rate", default=3e-4, type=float, help="Learning rate")
@click.option("--use-lora/--no-lora", default=True, help="Use LoRA")
@click.option("--use-4bit/--no-4bit", default=True, help="Use 4-bit quantization")
@click.option("--output-dir", default="./models/finetuned", help="Output directory")
@click.option("--push-to-hf", is_flag=True, help="Push to HuggingFace Hub")
@click.option("--hub-model-id", help="HuggingFace model ID")
@click.option("--deploy-sagemaker", is_flag=True, help="Deploy to SageMaker")
@click.option("--endpoint-name", help="SageMaker endpoint name")
@click.option("--dummy", is_flag=True, help="Run in dummy mode")
@click.option("--no-cache", is_flag=True, help="Disable pipeline caching")
def run_train(
    model: str,
    dataset: str,
    finetuning_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    use_lora: bool,
    use_4bit: bool,
    output_dir: str,
    push_to_hf: bool,
    hub_model_id: Optional[str],
    deploy_sagemaker: bool,
    endpoint_name: Optional[str],
    dummy: bool,
    no_cache: bool,
):
    """Run model training pipeline."""
    logger.info(f"Starting {finetuning_type} training pipeline")
    
    run_name = generate_run_name(f"training_{finetuning_type}")
    pipeline = model_training_pipeline.with_options(
        run_name=run_name,
        enable_cache=not no_cache,
    )
    
    result = pipeline(
        finetuning_type=finetuning_type,
        model_id=model,
        dataset_id=dataset,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        use_lora=use_lora,
        use_4bit=use_4bit,
        output_dir=output_dir,
        push_to_hub=push_to_hf,
        hub_model_id=hub_model_id,
        deploy_to_sagemaker_flag=deploy_sagemaker,
        endpoint_name=endpoint_name,
        is_dummy=dummy,
    )
    logger.success(f"Training completed: {result}")


@run.command(name="evaluate")
@click.option("--model", required=True, help="Model ID to evaluate")
@click.option("--dataset", help="Evaluation dataset ID")
@click.option("--metrics", multiple=True, default=["g-eval", "accuracy"], help="Metrics to compute")
@click.option("--benchmarks", multiple=True, help="Benchmarks to run (mmlu, gsm8k)")
@click.option("--eval-samples", default=100, type=int, help="Number of eval samples")
@click.option("--benchmark-samples", default=100, type=int, help="Number of benchmark samples")
@click.option("--no-cache", is_flag=True, help="Disable pipeline caching")
def run_evaluate(
    model: str,
    dataset: Optional[str],
    metrics: tuple,
    benchmarks: tuple,
    eval_samples: int,
    benchmark_samples: int,
    no_cache: bool,
):
    """Run model evaluation pipeline."""
    logger.info("Starting model evaluation pipeline")
    
    run_name = generate_run_name("model_evaluation")
    pipeline = model_evaluation_pipeline.with_options(
        run_name=run_name,
        enable_cache=not no_cache,
    )
    
    result = pipeline(
        model_id=model,
        eval_dataset_id=dataset,
        metrics=list(metrics),
        benchmarks=list(benchmarks) if benchmarks else None,
        num_eval_samples=eval_samples,
        num_benchmark_samples=benchmark_samples,
    )
    logger.success(f"Evaluation completed: {result}")


if __name__ == "__main__":
    cli()
