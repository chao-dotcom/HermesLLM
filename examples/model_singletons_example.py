"""
Model Singletons Example

This example demonstrates how to use the thread-safe singleton pattern for
efficient model management in HermesLLM. Singletons ensure that models are
loaded only once in memory, even when instantiated multiple times.
"""

import numpy as np
from loguru import logger

from hermes.models import EmbeddingModelSingleton, CrossEncoderModelSingleton


def example_embedding_singleton():
    """
    Demonstrate EmbeddingModelSingleton usage.
    
    The singleton pattern ensures that even when you create multiple instances,
    they all reference the same underlying model in memory.
    """
    logger.info("=" * 60)
    logger.info("Example 1: EmbeddingModelSingleton")
    logger.info("=" * 60)
    
    # First instantiation loads the model
    embedder1 = EmbeddingModelSingleton(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    logger.info(f"Embedder 1 ID: {id(embedder1)}")
    logger.info(f"Model ID: {embedder1.model_id}")
    logger.info(f"Embedding size: {embedder1.embedding_size}")
    logger.info(f"Max input length: {embedder1.max_input_length}")
    
    # Second instantiation returns the same instance
    embedder2 = EmbeddingModelSingleton(
        model_id="different-model",  # This is ignored!
        device="cuda"  # This is also ignored!
    )
    logger.info(f"Embedder 2 ID: {id(embedder2)}")
    
    # Verify they are the same instance
    assert embedder1 is embedder2, "Should be the same instance!"
    logger.success("✓ Both embedders are the same instance")
    
    # Generate single embedding
    text = "This is a test sentence for embedding generation."
    embedding = embedder1(text)
    logger.info(f"Single embedding shape: {len(embedding)}")
    
    # Generate batch embeddings
    texts = [
        "First document about machine learning.",
        "Second document about deep learning.",
        "Third document about neural networks."
    ]
    embeddings = embedder1(texts, show_progress_bar=False)
    logger.info(f"Batch embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
    
    # Get numpy array instead of list
    np_embeddings = embedder1(texts, to_list=False)
    logger.info(f"NumPy embeddings shape: {np_embeddings.shape}")
    logger.info(f"NumPy embeddings dtype: {np_embeddings.dtype}")
    
    logger.success("✓ EmbeddingModelSingleton example completed")


def example_crossencoder_singleton():
    """
    Demonstrate CrossEncoderModelSingleton usage.
    
    Cross-encoders are used for reranking query-document pairs by computing
    relevance scores. The singleton ensures efficient memory usage.
    """
    logger.info("=" * 60)
    logger.info("Example 2: CrossEncoderModelSingleton")
    logger.info("=" * 60)
    
    # First instantiation loads the model
    reranker1 = CrossEncoderModelSingleton(
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )
    logger.info(f"Reranker 1 ID: {id(reranker1)}")
    logger.info(f"Model ID: {reranker1.model_id}")
    
    # Second instantiation returns the same instance
    reranker2 = CrossEncoderModelSingleton()
    logger.info(f"Reranker 2 ID: {id(reranker2)}")
    
    # Verify singleton behavior
    assert reranker1 is reranker2, "Should be the same instance!"
    logger.success("✓ Both rerankers are the same instance")
    
    # Score query-document pairs
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks for complex tasks.",
        "The weather is nice today."
    ]
    
    pairs = [(query, doc) for doc in documents]
    scores = reranker1(pairs, show_progress_bar=False)
    
    logger.info(f"Number of pairs scored: {len(scores)}")
    for doc, score in zip(documents, scores):
        logger.info(f"Score: {score:.4f} | {doc[:50]}...")
    
    # Sort by relevance
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    logger.info("\nRanked documents (most relevant first):")
    for i, (doc, score) in enumerate(ranked, 1):
        logger.info(f"{i}. [{score:.4f}] {doc}")
    
    logger.success("✓ CrossEncoderModelSingleton example completed")


def example_memory_efficiency():
    """
    Demonstrate memory efficiency of singleton pattern.
    
    Without singletons, each instance would load a separate copy of the model,
    consuming significant memory. With singletons, all instances share the same
    model in memory.
    """
    logger.info("=" * 60)
    logger.info("Example 3: Memory Efficiency")
    logger.info("=" * 60)
    
    # Create multiple "instances" (all reference the same model)
    embedders = [
        EmbeddingModelSingleton() for _ in range(5)
    ]
    
    # Verify all are the same instance
    all_same = all(embedder is embedders[0] for embedder in embedders)
    logger.info(f"Created {len(embedders)} embedder 'instances'")
    logger.info(f"All reference same object: {all_same}")
    logger.success(f"✓ Memory saved: Only 1 model loaded instead of {len(embedders)}")
    
    # All instances can be used interchangeably
    text = "Test sentence"
    results = [embedder(text) for embedder in embedders]
    
    # All produce identical results
    all_identical = all(
        np.array_equal(result, results[0]) for result in results
    )
    logger.info(f"All results identical: {all_identical}")
    logger.success("✓ Singleton pattern ensures consistency")


def example_cached_properties():
    """
    Demonstrate cached property usage.
    
    The embedding_size property is computed once and cached for efficiency.
    """
    logger.info("=" * 60)
    logger.info("Example 4: Cached Properties")
    logger.info("=" * 60)
    
    embedder = EmbeddingModelSingleton()
    
    # First access computes the value
    logger.info("Accessing embedding_size for the first time...")
    size1 = embedder.embedding_size
    logger.info(f"Embedding size: {size1}")
    
    # Subsequent accesses use cached value (no computation)
    logger.info("Accessing embedding_size again (cached)...")
    size2 = embedder.embedding_size
    logger.info(f"Embedding size: {size2}")
    
    assert size1 == size2
    logger.success("✓ Cached property works correctly")
    
    # Other properties
    logger.info(f"Max input length: {embedder.max_input_length}")
    logger.info(f"Tokenizer: {type(embedder.tokenizer).__name__}")


def example_integration_with_pipeline():
    """
    Demonstrate how singletons integrate with processing pipelines.
    
    When multiple components use the same model, they share the singleton instance.
    """
    logger.info("=" * 60)
    logger.info("Example 5: Pipeline Integration")
    logger.info("=" * 60)
    
    # Simulate multiple pipeline components
    class EmbeddingComponent1:
        def __init__(self):
            self.embedder = EmbeddingModelSingleton()
        
        def process(self, text):
            return self.embedder(text)
    
    class EmbeddingComponent2:
        def __init__(self):
            self.embedder = EmbeddingModelSingleton()
        
        def process(self, texts):
            return self.embedder(texts)
    
    # Create components
    comp1 = EmbeddingComponent1()
    comp2 = EmbeddingComponent2()
    
    # Verify they share the same model
    assert comp1.embedder is comp2.embedder
    logger.success("✓ Components share the same model instance")
    
    # Use components
    text = "Shared model usage example"
    result1 = comp1.process(text)
    result2 = comp2.process([text])[0]
    
    assert np.array_equal(result1, result2)
    logger.success("✓ Results are consistent across components")


def example_device_management():
    """
    Demonstrate device management (CPU/GPU).
    
    Note: The device is set during first instantiation and cannot be changed
    for the singleton instance.
    """
    logger.info("=" * 60)
    logger.info("Example 6: Device Management")
    logger.info("=" * 60)
    
    # First instantiation sets device
    embedder = EmbeddingModelSingleton(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    logger.info(f"Device: {embedder.device}")
    
    # Generate embedding (runs on CPU)
    text = "Device management example"
    embedding = embedder(text)
    logger.info(f"Generated embedding of size: {len(embedding)}")
    
    # Note: To use GPU, instantiate with device="cuda" on first call
    # embedder = EmbeddingModelSingleton(device="cuda")  # First call only
    
    logger.success("✓ Device management example completed")


def example_error_handling():
    """
    Demonstrate error handling in singletons.
    
    Singletons gracefully handle errors and return empty results.
    """
    logger.info("=" * 60)
    logger.info("Example 7: Error Handling")
    logger.info("=" * 60)
    
    embedder = EmbeddingModelSingleton()
    
    # Valid input
    valid_text = "This is valid text"
    result = embedder(valid_text)
    logger.info(f"Valid input result length: {len(result)}")
    
    # Empty string (should work)
    empty_result = embedder("")
    logger.info(f"Empty string result length: {len(empty_result)}")
    
    # Batch with mixed content
    mixed_texts = [
        "Normal text",
        "",
        "Another normal text"
    ]
    batch_results = embedder(mixed_texts)
    logger.info(f"Batch results: {len(batch_results)} embeddings")
    
    logger.success("✓ Error handling works correctly")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 60)
    logger.info("MODEL SINGLETONS - COMPREHENSIVE EXAMPLES")
    logger.info("=" * 60 + "\n")
    
    # Run examples
    example_embedding_singleton()
    print()
    
    example_crossencoder_singleton()
    print()
    
    example_memory_efficiency()
    print()
    
    example_cached_properties()
    print()
    
    example_integration_with_pipeline()
    print()
    
    example_device_management()
    print()
    
    example_error_handling()
    print()
    
    logger.info("=" * 60)
    logger.success("All examples completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
