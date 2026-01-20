"""
ZenML Steps Module

This module contains ZenML step definitions that are composed
into pipelines.
"""

# Collection steps
from hermes.zenml_steps.collection_steps import (
    create_or_get_author,
    collect_from_links,
    collect_from_platform,
)

# Processing steps
from hermes.zenml_steps.processing_steps import (
    query_documents_from_db,
    clean_documents,
    chunk_documents,
    embed_chunks,
    load_to_vector_db,
    save_to_database,
)

# Dataset steps
from hermes.zenml_steps.dataset_steps import (
    query_cleaned_documents,
    generate_instruction_dataset,
    generate_preference_dataset,
    push_to_huggingface,
)

# Training steps
from hermes.zenml_steps.training_steps import (
    train_model,
    deploy_to_sagemaker,
)

# Evaluation steps
from hermes.zenml_steps.evaluation_steps import (
    evaluate_model,
    run_benchmarks,
)

# Export steps
from hermes.zenml_steps.export_steps import (
    export_to_json,
    collect_pipeline_artifacts,
)

__all__ = [
    # Collection
    "create_or_get_author",
    "collect_from_links",
    "collect_from_platform",
    # Processing
    "query_documents_from_db",
    "clean_documents",
    "chunk_documents",
    "embed_chunks",
    "load_to_vector_db",
    "save_to_database",
    # Dataset
    "query_cleaned_documents",
    "generate_instruction_dataset",
    "generate_preference_dataset",
    "push_to_huggingface",
    # Training
    "train_model",
    "deploy_to_sagemaker",
    # Evaluation
    "evaluate_model",
    "run_benchmarks",
    # Export
    "export_to_json",
    "collect_pipeline_artifacts",
]
