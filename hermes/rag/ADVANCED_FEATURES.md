# Advanced RAG Features

Self-query extraction, metadata enrichment, and multi-query expansion for enhanced retrieval.

## Quick Start

### Author Extraction

```python
from hermes.rag import AuthorExtractor
from hermes.core import Query

extractor = AuthorExtractor(api_key="your-openai-key")
query = Query(content="I am John Smith. Find my documents")
query = extractor.extract(query)

print(query.metadata["author_name"])  # "John Smith"
```

### Self-Query Metadata

```python
from hermes.rag import SelfQueryExtractor

extractor = SelfQueryExtractor(
    api_key="your-key",
    extract_author=True,
    extract_category=True,
    extract_intent=True,
)

query = Query(content="Write a blog post about AI safety")
query = extractor.extract(query)

print(query.metadata["category"])  # "AI safety"
print(query.metadata["intent"])    # "generate"
```

### Multi-Query Expansion

```python
from hermes.rag import MultiQueryExpander

expander = MultiQueryExpander(api_key="your-key", num_queries=3)
query = Query(content="What are advanced RAG methods?")
queries = expander.expand(query)

# Returns: [original, alternative1, alternative2]
```

### Enhanced RAG Pipeline

```python
from hermes.rag import RAGPipeline, SelfQueryExtractor, MultiQueryExpander

pipeline = RAGPipeline(
    self_query_extractor=SelfQueryExtractor(api_key="your-key"),
    query_expander=MultiQueryExpander(api_key="your-key"),
)

response = pipeline.query(
    "I am Sarah. What are the latest LLM techniques?",
    use_self_query=True,
    use_query_expansion=True,
)
```

## Features

| Feature | Purpose | Cost/Query |
|---------|---------|------------|
| **AuthorExtractor** | Extract user/author info | ~$0.0001 |
| **SelfQueryExtractor** | Extract category, intent, author | ~$0.0002 |
| **MetadataEnricher** | Add keywords, complexity score | Free |
| **MultiQueryExpander** | Generate query variations | ~$0.0003 |

## Components

### 1. Self-Query Extraction

Extracts structured metadata from natural language:
- Author name or ID
- Category/topic
- User intent (search, generate, summarize, etc.)

### 2. Author Extraction

Specialized extractor for author information:
- Faster than full self-query
- Supports names and numeric IDs
- Enables user-specific filtering

### 3. Metadata Enrichment

Adds computed metadata:
- Query length and word count
- Top keywords
- Complexity score (0-1)

### 4. Multi-Query Expansion

Generates multiple query perspectives:
- Original + N-1 alternatives
- Improves retrieval recall
- Overcomes semantic search limitations

## Use Cases

### 1. **Personalized Retrieval**
```python
# Extract author to filter documents
query = author_extractor.extract(query)
author = query.metadata["author_name"]
# Filter by author in vector DB
```

### 2. **Intent-Based Routing**
```python
# Route based on intent
query = self_query.extract(query)
if query.metadata["intent"] == "summarize":
    # Use summarization pipeline
elif query.metadata["intent"] == "search":
    # Use search pipeline
```

### 3. **Better Retrieval**
```python
# Expand query for better recall
queries = expander.expand(query)
# Retrieve with all variations
# Merge and deduplicate results
```

## Installation

```bash
pip install openai  # Required for self-query and expansion
```

## Examples

See [examples/advanced_rag_example.py](../examples/advanced_rag_example.py) for:

1. Author extraction
2. Full self-query extraction
3. Metadata enrichment
4. Multi-query expansion
5. Enhanced RAG pipeline
6. Author-filtered retrieval
7. Complex query workflow
8. Integration examples

## Documentation

Full documentation: [docs/ADVANCED_RAG.md](../docs/ADVANCED_RAG.md)

Topics:
- Detailed API reference
- Best practices
- Cost optimization
- Error handling
- Advanced patterns

## Architecture

```
Query Flow with All Features:

Raw Query
    ↓
Author Extraction → metadata["author_name"]
    ↓
Self-Query Extraction → metadata["category", "intent"]
    ↓
Metadata Enrichment → metadata["keywords", "complexity"]
    ↓
Query Expansion → [query1, query2, query3]
    ↓
Embed Each Query
    ↓
Retrieve Documents
    ↓
Merge & Deduplicate
    ↓
Rerank
    ↓
Generate Response
```

## Configuration

### Minimal (Author Only)
```python
pipeline = RAGPipeline(
    author_extractor=AuthorExtractor(api_key=api_key),
)

response = pipeline.query(query, use_author_extraction=True)
```

### Full Features
```python
pipeline = RAGPipeline(
    author_extractor=AuthorExtractor(api_key=api_key),
    self_query_extractor=SelfQueryExtractor(api_key=api_key),
    metadata_enricher=MetadataEnricher(),
    query_expander=MultiQueryExpander(api_key=api_key, num_queries=3),
)

response = pipeline.query(
    query,
    use_author_extraction=True,
    use_self_query=True,
    use_metadata_enrichment=True,
    use_query_expansion=True,
)
```

## Performance

- **Author Extraction**: ~100-200ms per query
- **Self-Query Extraction**: ~200-300ms per query
- **Metadata Enrichment**: <1ms per query (local)
- **Multi-Query Expansion**: ~300-500ms per query

## Best Practices

1. **Use AuthorExtractor** when you only need author info (faster)
2. **Limit expansion** to 3-4 queries for best cost/benefit
3. **Cache extractions** for repeated queries
4. **Validate metadata** before using (may be null)
5. **Monitor costs** with high-volume usage

## License

Same as parent project.
