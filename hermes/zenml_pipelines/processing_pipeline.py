"""
Processing Pipeline

ZenML pipeline for processing, cleaning, chunking, and embedding documents.
"""

from typing import List, Optional

from zenml import pipeline

from hermes.zenml_steps.processing_steps import (
    query_documents_from_db,
    clean_documents,
    chunk_documents,
    embed_chunks,
    load_to_vector_db,
    save_to_database,
)


@pipeline(name="document_processing_pipeline")
def document_processing_pipeline(
    author_names: List[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk: int = 256,
    collection_name: str = "hermes_documents",
    wait_for: Optional[str] = None,
) -> List[str]:
    """
    Pipeline for processing documents through cleaning, chunking, and embedding.
    
    This pipeline:
    1. Queries documents from MongoDB
    2. Cleans documents (removes HTML, URLs, fixes unicode)
    3. Saves cleaned documents back to database
    4. Chunks cleaned documents using token-aware splitting
    5. Generates embeddings for chunks
    6. Loads embedded chunks to vector database
    
    Args:
        author_names: Optional list of author names to filter
        model_name: Embedding model name
        tokens_per_chunk: Maximum tokens per chunk
        collection_name: Vector DB collection name
        wait_for: Optional step ID to wait for (for orchestration)
        
    Returns:
        List of step invocation IDs
    """
    # Step 1: Query raw documents from database
    documents = query_documents_from_db(
        author_names=author_names,
        after=wait_for,
    )
    
    # Step 2: Clean documents
    cleaned_documents = clean_documents(documents=documents)
    
    # Step 3: Save cleaned documents to database
    save_status = save_to_database(cleaned_documents=cleaned_documents)
    
    # Step 4: Chunk documents
    chunks = chunk_documents(
        cleaned_documents=cleaned_documents,
        model_name=model_name,
        tokens_per_chunk=tokens_per_chunk,
    )
    
    # Step 5: Embed chunks
    embedded_chunks = embed_chunks(
        chunks=chunks,
        model_name=model_name,
    )
    
    # Step 6: Load to vector database
    load_status = load_to_vector_db(
        embedded_chunks=embedded_chunks,
        collection_name=collection_name,
    )
    
    # Return step IDs for orchestration
    return [save_status.invocation_id, load_status.invocation_id]
