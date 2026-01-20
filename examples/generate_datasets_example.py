"""
Example: Generate Training Datasets with AI
=============================================

This example demonstrates how to use the AI-powered dataset generation system
to create instruction-following and preference datasets from your documents.
"""

import os
from loguru import logger

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

from hermes.core import CleanedDocument, DataCategory
from hermes.datasets.generators import InstructionDatasetGenerator, PreferenceDatasetGenerator
from hermes.training.dataset_builder import InstructDatasetBuilder, PreferenceDatasetBuilder


def example_instruction_generation():
    """Generate instruction-following dataset."""
    logger.info("=== Instruction Dataset Generation ===")
    
    # Create some example cleaned documents
    documents = [
        CleanedDocument(
            content="""Machine learning is a subset of artificial intelligence that focuses on 
            building systems that learn from data. Instead of being explicitly programmed,
            these systems improve their performance through experience.""",
            platform="medium",
            author_id="user123",
            type="article",
            metadata={"title": "Intro to ML"}
        ),
        CleanedDocument(
            content="""Fine-tuning is the process of taking a pre-trained model and training it 
            further on a specific dataset. This allows the model to adapt to a particular task 
            while retaining the general knowledge it gained during initial training.""",
            platform="linkedin",
            author_id="user123",
            type="post",
            metadata={"title": "Fine-tuning Explained"}
        ),
    ]
    
    # Initialize builder
    builder = InstructDatasetBuilder()
    
    # Generate datasets using AI (set mock=True for testing without API calls)
    datasets = builder.generate_from_documents(
        documents=documents,
        test_size=0.2,
        mock=True  # Set to False to use real OpenAI API
    )
    
    # Access train and test sets
    train_datasets = datasets["train"]
    test_datasets = datasets["test"]
    
    logger.info(f"Generated {sum(d.num_samples for d in train_datasets.values())} training samples")
    logger.info(f"Generated {sum(d.num_samples for d in test_datasets.values())} test samples")
    
    # Save to file
    builder.save_to_file(datasets, "instruction_dataset.json")
    logger.info("Saved datasets to instruction_dataset.json")
    
    return datasets


def example_preference_generation():
    """Generate preference/DPO dataset."""
    logger.info("=== Preference Dataset Generation ===")
    
    documents = [
        CleanedDocument(
            content="""RAG (Retrieval-Augmented Generation) combines retrieval and generation. 
            It searches a knowledge base for relevant information, then uses that context to 
            generate accurate responses. This helps reduce hallucinations and keeps answers 
            grounded in real data.""",
            platform="medium",
            author_id="user123",
            type="article",
            metadata={"title": "Understanding RAG"}
        ),
    ]
    
    # Initialize builder
    builder = PreferenceDatasetBuilder()
    
    # Generate datasets
    datasets = builder.generate_from_documents(
        documents=documents,
        test_size=0.2,
        mock=True  # Set to False to use real OpenAI API
    )
    
    logger.info(f"Generated preference dataset with train/test splits")
    
    # Save to file
    builder.save_to_file(datasets, "preference_dataset.json")
    logger.info("Saved datasets to preference_dataset.json")
    
    return datasets


def example_direct_generator_usage():
    """Use generators directly for more control."""
    logger.info("=== Direct Generator Usage ===")
    
    # Create documents
    documents = [
        CleanedDocument(
            content="Neural networks are computing systems inspired by biological neural networks.",
            platform="medium",
            author_id="user123",
            type="article",
        ),
    ]
    
    # Initialize generator
    generator = InstructionDatasetGenerator(
        openai_model="gpt-4o-mini"
    )
    
    # Create prompts
    prompts = generator.get_prompts(documents, extract_chunks=False)
    
    # Generate datasets
    datasets = generator.generate(
        prompts=prompts,
        test_size=0.2,
        mock=True
    )
    
    logger.info("Generated datasets using direct generator API")
    
    return datasets


if __name__ == "__main__":
    # Run examples
    example_instruction_generation()
    example_preference_generation()
    example_direct_generator_usage()
