"""
Advanced RAG examples with self-query and metadata enrichment.

Demonstrates:
1. Author extraction from queries
2. Full self-query metadata extraction
3. Query metadata enrichment
4. Multi-query expansion
5. Enhanced RAG pipeline with all features
"""

import os
from loguru import logger

from hermes.core import Query
from hermes.rag import (
    SelfQueryExtractor,
    AuthorExtractor,
    MetadataEnricher,
    MultiQueryExpander,
    RAGPipeline,
)


# Example 1: Author Extraction
def example_author_extraction():
    """Extract author information from queries."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize extractor
    author_extractor = AuthorExtractor(api_key=api_key)
    
    # Test queries
    queries = [
        "I am John Smith and I want to write about AI safety",
        "My user ID is 12345. Search for machine learning papers",
        "What are the best RAG techniques?",  # No author
        "Paul Iusztin here. Generate a blog post about LLMs",
    ]
    
    print("\n" + "="*60)
    print("AUTHOR EXTRACTION EXAMPLES")
    print("="*60)
    
    for query_text in queries:
        query = Query(content=query_text)
        enriched_query = author_extractor.extract(query)
        
        print(f"\nQuery: {query_text}")
        if enriched_query.metadata.get("author_name"):
            print(f"  Author Name: {enriched_query.metadata['author_name']}")
            print(f"  First Name: {enriched_query.metadata.get('author_first_name')}")
            print(f"  Last Name: {enriched_query.metadata.get('author_last_name')}")
        elif enriched_query.metadata.get("author_id"):
            print(f"  Author ID: {enriched_query.metadata['author_id']}")
        else:
            print("  No author found")


# Example 2: Full Self-Query Extraction
def example_self_query_extraction():
    """Extract comprehensive metadata from queries."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize with all extraction enabled
    extractor = SelfQueryExtractor(
        api_key=api_key,
        extract_author=True,
        extract_category=True,
        extract_intent=True,
    )
    
    # Test queries
    queries = [
        "I am Alice Johnson. Write a blog post about quantum computing",
        "My user ID is 99999. Search for papers on reinforcement learning",
        "Summarize the latest research on transformers",
        "Generate an article about climate change for my blog",
    ]
    
    print("\n" + "="*60)
    print("FULL SELF-QUERY EXTRACTION")
    print("="*60)
    
    for query_text in queries:
        query = Query(content=query_text)
        enriched_query = extractor.extract(query)
        
        print(f"\nQuery: {query_text}")
        print(f"  Metadata:")
        for key, value in enriched_query.metadata.items():
            if key != "extracted_metadata":  # Skip raw data
                print(f"    {key}: {value}")


# Example 3: Metadata Enrichment
def example_metadata_enrichment():
    """Enrich queries with computed metadata."""
    
    enricher = MetadataEnricher(
        extract_keywords=True,
        compute_complexity=True,
    )
    
    queries = [
        "What is RAG?",
        "Explain the differences between supervised and unsupervised learning in machine learning",
        "How do transformers work? What are attention mechanisms? Why are they important?",
    ]
    
    print("\n" + "="*60)
    print("METADATA ENRICHMENT")
    print("="*60)
    
    for query_text in queries:
        query = Query(content=query_text)
        enriched = enricher.enrich(query)
        
        print(f"\nQuery: {query_text}")
        print(f"  Length: {enriched.metadata['query_length']}")
        print(f"  Words: {enriched.metadata['word_count']}")
        print(f"  Keywords: {enriched.metadata['keywords']}")
        print(f"  Complexity: {enriched.metadata['complexity_score']}")


# Example 4: Multi-Query Expansion
def example_multi_query_expansion():
    """Expand queries into multiple perspectives."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize expander
    expander = MultiQueryExpander(
        api_key=api_key,
        num_queries=4,  # Original + 3 expansions
    )
    
    query = Query(content="What are the best types of advanced RAG methods?")
    
    print("\n" + "="*60)
    print("MULTI-QUERY EXPANSION")
    print("="*60)
    print(f"\nOriginal: {query.content}")
    print("\nExpanded queries:")
    
    expanded = expander.expand(query)
    for i, q in enumerate(expanded, 1):
        print(f"{i}. {q.content}")
        if q.metadata.get("is_expansion"):
            print(f"   (expansion of: {q.metadata['original_query'][:50]}...)")


# Example 5: Enhanced RAG Pipeline
def example_enhanced_rag_pipeline():
    """Complete RAG pipeline with self-query features."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize components
    author_extractor = AuthorExtractor(api_key=api_key)
    self_query_extractor = SelfQueryExtractor(api_key=api_key)
    metadata_enricher = MetadataEnricher()
    multi_query_expander = MultiQueryExpander(api_key=api_key, num_queries=3)
    
    # Create enhanced pipeline
    pipeline = RAGPipeline(
        author_extractor=author_extractor,
        self_query_extractor=self_query_extractor,
        metadata_enricher=metadata_enricher,
        query_expander=multi_query_expander,
    )
    
    # Query with all features enabled
    query_text = "I am Sarah Chen. What are the latest advances in LLM fine-tuning?"
    
    print("\n" + "="*60)
    print("ENHANCED RAG PIPELINE")
    print("="*60)
    print(f"\nQuery: {query_text}")
    
    # This will:
    # 1. Extract author (Sarah Chen)
    # 2. Extract category/intent
    # 3. Enrich with metadata (length, keywords, complexity)
    # 4. Expand to multiple queries
    # 5. Retrieve and rerank
    # 6. Generate response
    response = pipeline.query(
        query_text,
        use_author_extraction=True,
        use_self_query=True,
        use_metadata_enrichment=True,
        use_query_expansion=True,
    )
    
    print(f"\nResponse: {response}")


# Example 6: Author-Filtered Retrieval
def example_author_filtered_retrieval():
    """Use author info to filter documents."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Extract author
    author_extractor = AuthorExtractor(api_key=api_key)
    
    query = Query(content="I am Paul Iusztin. Show me my articles about RAG")
    query = author_extractor.extract(query)
    
    print("\n" + "="*60)
    print("AUTHOR-FILTERED RETRIEVAL")
    print("="*60)
    print(f"\nQuery: {query.content}")
    
    if query.metadata.get("author_name"):
        author_name = query.metadata["author_name"]
        print(f"Extracted Author: {author_name}")
        print(f"\nCan now filter documents to only show content by {author_name}")
        print("This enables personalized retrieval based on user identity")
    else:
        print("No author found - showing all results")


# Example 7: Complex Query with Full Pipeline
def example_complex_query_workflow():
    """Step-by-step workflow with all features."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Step 1: Start with raw query
    raw_query = "My name is Dr. Jane Smith and my ID is 42. Write a technical post about DPO vs RLHF for fine-tuning LLMs"
    query = Query(content=raw_query)
    
    print("\n" + "="*60)
    print("COMPLEX QUERY WORKFLOW")
    print("="*60)
    print(f"\n[Step 1] Raw Query:\n{raw_query}")
    
    # Step 2: Extract author
    author_extractor = AuthorExtractor(api_key=api_key)
    query = author_extractor.extract(query)
    print(f"\n[Step 2] Author Extracted:")
    print(f"  Name: {query.metadata.get('author_name')}")
    print(f"  First: {query.metadata.get('author_first_name')}")
    print(f"  Last: {query.metadata.get('author_last_name')}")
    
    # Step 3: Extract full metadata
    self_query = SelfQueryExtractor(
        api_key=api_key,
        extract_author=False,  # Already extracted
        extract_category=True,
        extract_intent=True,
    )
    query = self_query.extract(query)
    print(f"\n[Step 3] Metadata Extracted:")
    print(f"  Category: {query.metadata.get('category')}")
    print(f"  Intent: {query.metadata.get('intent')}")
    
    # Step 4: Enrich with computed metadata
    enricher = MetadataEnricher()
    query = enricher.enrich(query)
    print(f"\n[Step 4] Enriched Metadata:")
    print(f"  Keywords: {query.metadata.get('keywords')}")
    print(f"  Complexity: {query.metadata.get('complexity_score')}")
    
    # Step 5: Expand query
    expander = MultiQueryExpander(api_key=api_key, num_queries=3)
    queries = expander.expand(query)
    print(f"\n[Step 5] Query Expansion:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q.content}")
    
    print(f"\n[Final] Enriched Query Metadata:")
    import json
    print(json.dumps(query.metadata, indent=2))


# Example 8: Integration with Existing RAG
def example_integration_with_existing_rag():
    """Show how to add self-query to existing RAG setup."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("\n" + "="*60)
    print("INTEGRATION EXAMPLE")
    print("="*60)
    
    # Existing RAG pipeline
    print("\n[Before] Standard RAG:")
    print("  query -> embed -> retrieve -> rerank -> generate")
    
    # Enhanced with self-query
    print("\n[After] Enhanced RAG:")
    print("  query -> extract_author -> extract_metadata -> enrich")
    print("        -> expand_query -> embed -> retrieve -> rerank -> generate")
    
    print("\n[Benefits]:")
    print("  ✓ Author-aware retrieval")
    print("  ✓ Category/intent-based filtering")
    print("  ✓ Better retrieval via query expansion")
    print("  ✓ Richer query understanding")
    
    # Code example
    print("\n[Code]:")
    code = '''
# Add to existing pipeline
from hermes.rag import AuthorExtractor, SelfQueryExtractor

pipeline = RAGPipeline(
    # ... existing components ...
    author_extractor=AuthorExtractor(api_key=api_key),
    self_query_extractor=SelfQueryExtractor(api_key=api_key),
)

# Use with flags
response = pipeline.query(
    "I am John. Find my documents about AI",
    use_author_extraction=True,
    use_self_query=True,
)
'''
    print(code)


def main():
    """Run all examples."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    examples = [
        ("Author Extraction", example_author_extraction),
        ("Self-Query Extraction", example_self_query_extraction),
        ("Metadata Enrichment", example_metadata_enrichment),
        ("Multi-Query Expansion", example_multi_query_expansion),
        ("Complex Workflow", example_complex_query_workflow),
        ("Author-Filtered Retrieval", example_author_filtered_retrieval),
        ("Integration Example", example_integration_with_existing_rag),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            logger.error(f"{name} failed: {e}")


if __name__ == "__main__":
    # Run specific example
    example_author_extraction()
    
    # Or run all
    # main()
