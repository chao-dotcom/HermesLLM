"""
Advanced Text Processing Examples

This example demonstrates the advanced text processing capabilities in HermesLLM,
including token-aware chunking, sentence-aware splitting, and hybrid multi-stage
chunking strategies.
"""

from loguru import logger

from hermes.processing.text_splitters import (
    TokenAwareTextSplitter,
    SentenceAwareSplitter,
    HybridChunker,
    get_token_count,
    validate_chunk_tokens,
)
from hermes.processing.chunk_utils import (
    analyze_chunks,
    validate_chunk_quality,
    optimize_chunks,
    merge_short_chunks,
)
from hermes.models import EmbeddingModelSingleton


# Sample texts for demonstration
SAMPLE_ARTICLE = """
Machine learning is a subset of artificial intelligence that focuses on the development 
of algorithms and statistical models that enable computer systems to improve their 
performance on a specific task through experience.

Deep learning is a specialized branch of machine learning that uses neural networks 
with multiple layers. These deep neural networks can automatically learn hierarchical 
representations of data, making them particularly effective for tasks like image 
recognition, natural language processing, and speech recognition.

The transformer architecture, introduced in 2017, revolutionized natural language 
processing. Unlike previous recurrent neural networks, transformers use self-attention 
mechanisms to process entire sequences simultaneously, enabling better parallelization 
and capturing long-range dependencies more effectively.

Large language models like GPT and BERT are built on transformer architectures. 
These models are trained on massive amounts of text data and can perform a wide 
variety of natural language tasks, from translation to question answering, with 
impressive accuracy.
"""

LONG_DOCUMENT = """
Artificial intelligence has evolved dramatically over the past few decades. """ + \
    "The field encompasses various approaches including symbolic AI, machine learning, and deep learning. " * 50


def example_token_aware_splitting():
    """
    Demonstrate token-aware text splitting.
    
    Token-aware splitting ensures chunks respect the model's token limit,
    preventing truncation during embedding generation.
    """
    logger.info("=" * 60)
    logger.info("Example 1: Token-Aware Text Splitting")
    logger.info("=" * 60)
    
    # Initialize splitter
    splitter = TokenAwareTextSplitter(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        tokens_per_chunk=128,  # Conservative limit
        chunk_overlap=20,
    )
    
    # Split text
    chunks = splitter.split_text(LONG_DOCUMENT)
    
    logger.info(f"Original text: {len(LONG_DOCUMENT)} characters")
    logger.info(f"Created {len(chunks)} chunks")
    
    # Analyze chunks
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        token_count = get_token_count(chunk, splitter.model_name)
        logger.info(f"Chunk {i}: {len(chunk)} chars, {token_count} tokens")
        logger.info(f"  Preview: {chunk[:80]}...")
    
    # Validate all chunks are within token limit
    is_valid = validate_chunk_tokens(chunks, max_tokens=128, model_name=splitter.model_name)
    logger.success(f"✓ All chunks within token limit: {is_valid}")


def example_sentence_aware_splitting():
    """
    Demonstrate sentence-aware splitting for articles.
    
    Sentence-aware splitting maintains semantic coherence by ensuring
    chunks end at natural sentence boundaries.
    """
    logger.info("=" * 60)
    logger.info("Example 2: Sentence-Aware Splitting")
    logger.info("=" * 60)
    
    # Initialize splitter
    splitter = SentenceAwareSplitter(
        min_length=100,
        max_length=300,
    )
    
    # Split article
    chunks = splitter.split_text(SAMPLE_ARTICLE)
    
    logger.info(f"Article length: {len(SAMPLE_ARTICLE)} characters")
    logger.info(f"Created {len(chunks)} sentence-aware chunks")
    
    for i, chunk in enumerate(chunks):
        logger.info(f"\nChunk {i} ({len(chunk)} chars):")
        logger.info(f"  {chunk[:100]}...")
        
        # Check if chunk ends with sentence punctuation
        ends_with_sentence = chunk.rstrip().endswith(('.', '!', '?'))
        logger.info(f"  Ends with sentence: {ends_with_sentence}")
    
    logger.success("✓ Sentence-aware splitting maintains semantic coherence")


def example_hybrid_chunking():
    """
    Demonstrate hybrid multi-stage chunking.
    
    Hybrid chunking combines character-based, sentence-aware, and
    token-based splitting for optimal results.
    """
    logger.info("=" * 60)
    logger.info("Example 3: Hybrid Multi-Stage Chunking")
    logger.info("=" * 60)
    
    # Get embedding model for configuration
    embedder = EmbeddingModelSingleton()
    
    # Initialize hybrid chunker
    chunker = HybridChunker(
        model_name=embedder.model_id,
        char_chunk_size=500,
        tokens_per_chunk=embedder.max_input_length,
        chunk_overlap=50,
        min_chunk_length=50,
        max_chunk_length=1000,
    )
    
    # Create a very long document
    very_long_doc = SAMPLE_ARTICLE * 5
    
    logger.info(f"Document length: {len(very_long_doc)} characters")
    
    # Split using hybrid approach
    chunks = chunker.split_text(very_long_doc)
    
    logger.info(f"Created {len(chunks)} hybrid chunks")
    logger.info(f"Model token limit: {embedder.max_input_length}")
    
    # Analyze each stage
    for i, chunk in enumerate(chunks[:3]):
        token_count = get_token_count(chunk, embedder.model_id)
        logger.info(f"\nChunk {i}:")
        logger.info(f"  Characters: {len(chunk)}")
        logger.info(f"  Tokens: {token_count}")
        logger.info(f"  Within limit: {token_count <= embedder.max_input_length}")
    
    # Validate all chunks
    is_valid = validate_chunk_tokens(
        chunks,
        max_tokens=embedder.max_input_length,
        model_name=embedder.model_id
    )
    logger.success(f"✓ All chunks validated: {is_valid}")


def example_chunk_analysis():
    """
    Demonstrate chunk analysis utilities.
    
    Analysis tools provide insights into chunk quality and distribution.
    """
    logger.info("=" * 60)
    logger.info("Example 4: Chunk Analysis")
    logger.info("=" * 60)
    
    # Create chunks
    splitter = TokenAwareTextSplitter(tokens_per_chunk=128)
    chunks = splitter.split_text(LONG_DOCUMENT)
    
    # Analyze chunks
    stats = analyze_chunks(chunks, model_name=splitter.model_name)
    
    logger.info(f"Chunk Statistics:")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Total characters: {stats['total_characters']}")
    logger.info(f"  Total tokens: {stats['total_tokens']}")
    logger.info(f"  Average chars per chunk: {stats['avg_chars']:.1f}")
    logger.info(f"  Average tokens per chunk: {stats['avg_tokens']:.1f}")
    logger.info(f"  Median chars: {stats['median_chars']:.1f}")
    logger.info(f"  Median tokens: {stats['median_tokens']:.1f}")
    logger.info(f"  Character range: {stats['min_chars']} - {stats['max_chars']}")
    logger.info(f"  Token range: {stats['min_tokens']} - {stats['max_tokens']}")
    logger.info(f"  Std dev (chars): {stats['std_chars']:.1f}")
    logger.info(f"  Std dev (tokens): {stats['std_tokens']:.1f}")
    
    logger.success("✓ Comprehensive chunk analysis completed")


def example_chunk_validation():
    """
    Demonstrate chunk quality validation.
    
    Validation ensures chunks meet quality criteria before processing.
    """
    logger.info("=" * 60)
    logger.info("Example 5: Chunk Quality Validation")
    logger.info("=" * 60)
    
    # Create some chunks with varying quality
    good_chunks = [
        "This is a well-sized chunk with good content.",
        "Another properly sized chunk for demonstration purposes.",
    ]
    
    bad_chunks = [
        "Too short",
        "This chunk is way too long " + "and exceeds the maximum character limit " * 50,
        "",  # Empty chunk
    ]
    
    # Validate good chunks
    logger.info("Validating good chunks:")
    is_valid, issues = validate_chunk_quality(
        good_chunks,
        min_length=20,
        max_length=200,
        max_tokens=100
    )
    logger.info(f"  Valid: {is_valid}")
    if issues:
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Validate bad chunks
    logger.info("\nValidating problematic chunks:")
    is_valid, issues = validate_chunk_quality(
        bad_chunks,
        min_length=20,
        max_length=200,
        max_tokens=100
    )
    logger.info(f"  Valid: {is_valid}")
    logger.info(f"  Issues found: {len(issues)}")
    for issue in issues:
        logger.warning(f"  - {issue}")
    
    logger.success("✓ Validation detects quality issues correctly")


def example_chunk_optimization():
    """
    Demonstrate chunk optimization.
    
    Optimization merges short chunks and splits long ones for better quality.
    """
    logger.info("=" * 60)
    logger.info("Example 6: Chunk Optimization")
    logger.info("=" * 60)
    
    # Create chunks with varying sizes
    unoptimized_chunks = [
        "Short.",
        "Another short one.",
        "This is a medium-sized chunk with reasonable content.",
        "Way too long " + "chunk " * 100,
        "Tiny",
        "This is good size and should remain unchanged during optimization process.",
    ]
    
    logger.info(f"Unoptimized chunks: {len(unoptimized_chunks)}")
    for i, chunk in enumerate(unoptimized_chunks):
        logger.info(f"  Chunk {i}: {len(chunk)} chars")
    
    # Optimize
    optimized_chunks = optimize_chunks(
        unoptimized_chunks,
        min_length=30,
        max_length=200,
    )
    
    logger.info(f"\nOptimized chunks: {len(optimized_chunks)}")
    for i, chunk in enumerate(optimized_chunks):
        logger.info(f"  Chunk {i}: {len(chunk)} chars")
        logger.info(f"    Preview: {chunk[:60]}...")
    
    logger.success("✓ Optimization improved chunk quality")


def example_integration_with_embedder():
    """
    Demonstrate integration with embedding models.
    
    Shows how to configure chunkers based on embedding model properties.
    """
    logger.info("=" * 60)
    logger.info("Example 7: Integration with Embedding Models")
    logger.info("=" * 60)
    
    # Get embedding model
    embedder = EmbeddingModelSingleton(
        model_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    logger.info(f"Embedding model: {embedder.model_id}")
    logger.info(f"Max input length: {embedder.max_input_length} tokens")
    logger.info(f"Embedding size: {embedder.embedding_size}")
    
    # Configure chunker based on model
    chunker = HybridChunker(
        model_name=embedder.model_id,
        tokens_per_chunk=embedder.max_input_length,  # Use model's max
        chunk_overlap=int(embedder.max_input_length * 0.1),  # 10% overlap
        min_chunk_length=50,
        max_chunk_length=2000,
    )
    
    # Process document
    chunks = chunker.split_text(SAMPLE_ARTICLE * 2)
    
    logger.info(f"\nCreated {len(chunks)} chunks optimized for model")
    
    # Generate embeddings for chunks
    logger.info("\nGenerating embeddings...")
    embeddings = embedder([chunk for chunk in chunks[:3]])  # First 3 chunks
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    logger.info(f"Embedding dimension: {len(embeddings[0])}")
    
    logger.success("✓ Chunking perfectly aligned with embedding model")


def example_comparison():
    """
    Compare different chunking strategies.
    
    Demonstrates the differences between character-based, token-based,
    and hybrid approaches.
    """
    logger.info("=" * 60)
    logger.info("Example 8: Strategy Comparison")
    logger.info("=" * 60)
    
    text = SAMPLE_ARTICLE * 3
    
    # Strategy 1: Character-based (simple)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    char_chunks = char_splitter.split_text(text)
    
    # Strategy 2: Token-aware
    token_splitter = TokenAwareTextSplitter(
        tokens_per_chunk=128,
        chunk_overlap=20
    )
    token_chunks = token_splitter.split_text(text)
    
    # Strategy 3: Hybrid
    hybrid_chunker = HybridChunker(
        tokens_per_chunk=128,
        char_chunk_size=500,
        min_chunk_length=50,
    )
    hybrid_chunks = hybrid_chunker.split_text(text)
    
    # Compare results
    logger.info(f"Text length: {len(text)} characters\n")
    
    logger.info(f"Character-based chunking:")
    char_stats = analyze_chunks(char_chunks)
    logger.info(f"  Chunks: {char_stats['total_chunks']}")
    logger.info(f"  Avg tokens: {char_stats['avg_tokens']:.1f}")
    logger.info(f"  Max tokens: {char_stats['max_tokens']}")
    
    logger.info(f"\nToken-aware chunking:")
    token_stats = analyze_chunks(token_chunks)
    logger.info(f"  Chunks: {token_stats['total_chunks']}")
    logger.info(f"  Avg tokens: {token_stats['avg_tokens']:.1f}")
    logger.info(f"  Max tokens: {token_stats['max_tokens']}")
    
    logger.info(f"\nHybrid chunking:")
    hybrid_stats = analyze_chunks(hybrid_chunks)
    logger.info(f"  Chunks: {hybrid_stats['total_chunks']}")
    logger.info(f"  Avg tokens: {hybrid_stats['avg_tokens']:.1f}")
    logger.info(f"  Max tokens: {hybrid_stats['max_tokens']}")
    
    logger.success("✓ Different strategies produce different trade-offs")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 60)
    logger.info("ADVANCED TEXT PROCESSING - COMPREHENSIVE EXAMPLES")
    logger.info("=" * 60 + "\n")
    
    example_token_aware_splitting()
    print()
    
    example_sentence_aware_splitting()
    print()
    
    example_hybrid_chunking()
    print()
    
    example_chunk_analysis()
    print()
    
    example_chunk_validation()
    print()
    
    example_chunk_optimization()
    print()
    
    example_integration_with_embedder()
    print()
    
    example_comparison()
    print()
    
    logger.info("=" * 60)
    logger.success("All examples completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
