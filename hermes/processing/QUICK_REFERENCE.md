# Advanced Text Processing - Quick Reference

## Overview

Token-aware, sentence-aware, and hybrid text chunking for optimal embedding quality.

## Quick Start

```python
from hermes.processing.text_splitters import HybridChunker
from hermes.models import EmbeddingModelSingleton

# Get embedding model
embedder = EmbeddingModelSingleton()

# Create hybrid chunker
chunker = HybridChunker(
    model_name=embedder.model_id,
    tokens_per_chunk=embedder.max_input_length,
    chunk_overlap=50
)

# Split text
chunks = chunker.split_text(document)
```

## Text Splitters

### TokenAwareTextSplitter

Splits based on token count, not characters.

```python
from hermes.processing.text_splitters import TokenAwareTextSplitter

splitter = TokenAwareTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk=256,      # Max tokens per chunk
    chunk_overlap=50           # Overlapping tokens
)

chunks = splitter.split_text(text)
```

**When to use**: When token limits are critical (embedding models).

### SentenceAwareSplitter

Splits at sentence boundaries for semantic coherence.

```python
from hermes.processing.text_splitters import SentenceAwareSplitter

splitter = SentenceAwareSplitter(
    min_length=100,    # Min chars
    max_length=500     # Max chars
)

chunks = splitter.split_text(article)
```

**When to use**: Articles, blog posts, documentation.

### HybridChunker

Multi-stage: character → token → length filtering.

```python
from hermes.processing.text_splitters import HybridChunker

chunker = HybridChunker(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    char_chunk_size=500,       # Stage 1: char limit
    tokens_per_chunk=256,      # Stage 2: token limit
    chunk_overlap=50,          # Overlap in tokens
    min_chunk_length=50,       # Min chars
    max_chunk_length=1000      # Max chars
)

chunks = chunker.split_text(document)
```

**When to use**: Production systems (recommended).

## Document Chunkers

Process `CleanedDocument` → `List[Chunk]`

### TokenAwareChunker

```python
from hermes.processing.chunkers import TokenAwareChunker

chunker = TokenAwareChunker(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk=256,
    chunk_overlap=50
)

chunks = chunker.handle(cleaned_document)
```

### SentenceAwareChunker

```python
from hermes.processing.chunkers import SentenceAwareChunker

chunker = SentenceAwareChunker(
    min_length=100,
    max_length=500
)

chunks = chunker.handle(article_document)
```

### HybridSmartChunker

```python
from hermes.processing.chunkers import HybridSmartChunker

chunker = HybridSmartChunker(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk=256,
    min_chunk_length=50,
    max_chunk_length=1000
)

chunks = chunker.handle(document)
```

## Utilities

### Analyze Chunks

```python
from hermes.processing.chunk_utils import analyze_chunks

stats = analyze_chunks(chunks, model_name="...")

# Returns:
# {
#     'total_chunks': 25,
#     'avg_tokens': 127.5,
#     'max_tokens': 128,
#     'min_tokens': 89,
#     'std_tokens': 15.2,
#     ...
# }
```

### Validate Quality

```python
from hermes.processing.chunk_utils import validate_chunk_quality

is_valid, issues = validate_chunk_quality(
    chunks,
    min_length=50,
    max_length=1000,
    max_tokens=512
)

if not is_valid:
    print(f"{len(issues)} quality issues found")
```

### Optimize Chunks

```python
from hermes.processing.chunk_utils import optimize_chunks

optimized = optimize_chunks(
    chunks,
    min_length=100,
    max_length=1000,
    max_tokens=512
)
```

### Token Counting

```python
from hermes.processing.text_splitters import get_token_count

token_count = get_token_count(
    text,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Validate Token Limits

```python
from hermes.processing.text_splitters import validate_chunk_tokens

is_valid = validate_chunk_tokens(
    chunks,
    max_tokens=256,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Strategy Comparison

| Strategy | Chunks | Avg Tokens | Max Tokens | Semantic Quality |
|----------|--------|------------|------------|------------------|
| Character (500 chars) | 20 | 135 | 248 ❌ | Medium |
| Token (128 tokens) | 22 | 127 | 128 ✅ | Medium |
| Sentence (100-500) | 18 | 115 | 140 ✅ | High |
| Hybrid | 21 | 125 | 128 ✅ | High ✅ |

## Integration Patterns

### Pattern 1: Auto-Configure from Model

```python
from hermes.models import EmbeddingModelSingleton
from hermes.processing.text_splitters import HybridChunker

embedder = EmbeddingModelSingleton()

chunker = HybridChunker(
    model_name=embedder.model_id,
    tokens_per_chunk=embedder.max_input_length,
    chunk_overlap=int(embedder.max_input_length * 0.1)  # 10%
)

chunks = chunker.split_text(text)
embeddings = embedder(chunks)
```

### Pattern 2: Validate Before Embedding

```python
from hermes.processing.text_splitters import validate_chunk_tokens

chunks = chunker.split_text(document)

is_valid = validate_chunk_tokens(
    chunks,
    max_tokens=embedder.max_input_length,
    model_name=embedder.model_id
)

if is_valid:
    embeddings = embedder(chunks)
else:
    # Re-chunk or handle error
    pass
```

### Pattern 3: Analyze-Optimize-Embed

```python
from hermes.processing.chunk_utils import analyze_chunks, optimize_chunks

# Initial chunking
chunks = chunker.split_text(text)

# Analyze
stats = analyze_chunks(chunks)
print(f"Avg: {stats['avg_tokens']}, Max: {stats['max_tokens']}")

# Optimize if needed
if stats['max_tokens'] > embedder.max_input_length:
    chunks = optimize_chunks(chunks, max_tokens=embedder.max_input_length)

# Embed
embeddings = embedder(chunks)
```

## Common Use Cases

### Use Case 1: Articles/Blog Posts

```python
chunker = SentenceAwareChunker(
    min_length=150,
    max_length=600
)
```

### Use Case 2: Technical Documentation

```python
chunker = HybridSmartChunker(
    tokens_per_chunk=200,
    min_chunk_length=100,
    max_chunk_length=800
)
```

### Use Case 3: Code Files

```python
from hermes.processing.chunkers import CodeChunker

chunker = CodeChunker(
    chunk_size=1000,
    chunk_overlap=100
)
```

### Use Case 4: Long Research Papers

```python
chunker = HybridChunker(
    tokens_per_chunk=256,
    chunk_overlap=50,      # Higher overlap for context
    min_chunk_length=200,
    max_chunk_length=1200
)
```

## Chunk Overlap Guide

| Content Type | Recommended Overlap | Rationale |
|--------------|-------------------|-----------|
| Technical docs | 5-10% | Low redundancy needed |
| Articles | 10-15% | Maintain context |
| Narratives | 15-20% | Story continuity |
| Code | 10-15% | Function context |

## Performance Tips

1. **Reuse Splitters**: Create once, use multiple times
2. **Batch Validation**: Validate all chunks at once
3. **Profile First**: Test strategies on sample data
4. **Monitor Stats**: Track avg/max tokens regularly
5. **Cache Tokenizers**: `TokenAwareTextSplitter` caches automatically

## Troubleshooting

### ❌ "Chunks exceed token limit"

```python
# Use token-aware splitting
splitter = TokenAwareTextSplitter(
    tokens_per_chunk=embedder.max_input_length
)
```

### ❌ "Poor chunk quality / semantic breaks"

```python
# Use sentence-aware or hybrid
splitter = SentenceAwareSplitter(min_length=100, max_length=500)
```

### ❌ "Too many small chunks"

```python
# Merge short chunks
from hermes.processing.chunk_utils import merge_short_chunks
merged = merge_short_chunks(chunks, min_length=100)
```

### ❌ "Inconsistent chunk sizes"

```python
# Use token-based for consistency
splitter = TokenAwareTextSplitter(tokens_per_chunk=256)
```

## Examples

See [examples/advanced_text_processing_example.py](../examples/advanced_text_processing_example.py) for:
- Token-aware splitting
- Sentence-aware splitting
- Hybrid chunking
- Chunk analysis
- Quality validation
- Optimization techniques
- Strategy comparison

## Full Documentation

See [docs/ADVANCED_TEXT_PROCESSING.md](../docs/ADVANCED_TEXT_PROCESSING.md) for complete documentation.
