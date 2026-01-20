"""
Processing Steps for ZenML Pipelines

This module contains ZenML steps for document processing, cleaning,
chunking, and embedding.
"""

from typing import List
from typing_extensions import Annotated

from loguru import logger
from zenml import get_step_context, step

from hermes.core import Document, CleanedDocument, Chunk, EmbeddedChunk
from hermes.storage.database import DatabaseClient
from hermes.storage.vector_store import VectorStoreClient
from hermes.processing.cleaners import TextCleaner
from hermes.processing.chunkers import HybridSmartChunker
from hermes.processing.embedders import SentenceTransformerEmbedder


@step
def query_documents_from_db(
    author_names: List[str] = None,
    limit: int = None,
) -> Annotated[List[Document], "documents"]:
    """
    Query documents from MongoDB.
    
    Args:
        author_names: Optional list of author names to filter
        limit: Optional limit on number of documents
        
    Returns:
        List of documents
    """
    logger.info(f"Querying documents from database")
    
    try:
        db_client = DatabaseClient()
        
        # Build query filter
        query_filter = {}
        if author_names:
            query_filter["author_full_name"] = {"$in": author_names}
        
        # Query documents
        documents = db_client.find_documents(
            filter=query_filter,
            limit=limit
        )
        
        logger.success(f"Retrieved {len(documents)} documents from database")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="documents",
            metadata={
                "total_documents": len(documents),
                "author_filter": author_names or "all",
                "limit": limit or "none",
            }
        )
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to query documents: {e}")
        return []


@step
def clean_documents(
    documents: List[Document],
    remove_html: bool = True,
    remove_urls: bool = True,
    fix_unicode: bool = True,
) -> Annotated[List[CleanedDocument], "cleaned_documents"]:
    """
    Clean documents using text cleaner.
    
    Args:
        documents: List of raw documents
        remove_html: Whether to remove HTML tags
        remove_urls: Whether to remove URLs
        fix_unicode: Whether to fix unicode issues
        
    Returns:
        List of cleaned documents
    """
    logger.info(f"Cleaning {len(documents)} documents")
    
    cleaner = TextCleaner(
        remove_html=remove_html,
        remove_urls=remove_urls,
        fix_unicode=fix_unicode,
    )
    
    cleaned_documents = []
    for doc in documents:
        try:
            cleaned = cleaner.handle(doc)
            cleaned_documents.extend(cleaned)
        except Exception as e:
            logger.error(f"Failed to clean document {doc.id}: {e}")
    
    logger.success(f"Cleaned {len(cleaned_documents)} documents")
    
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="cleaned_documents",
        metadata={
            "input_documents": len(documents),
            "output_documents": len(cleaned_documents),
            "remove_html": remove_html,
            "remove_urls": remove_urls,
        }
    )
    
    return cleaned_documents


@step
def chunk_documents(
    cleaned_documents: List[CleanedDocument],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk: int = 256,
    min_chunk_length: int = 50,
) -> Annotated[List[Chunk], "chunks"]:
    """
    Chunk documents using hybrid smart chunker.
    
    Args:
        cleaned_documents: List of cleaned documents
        model_name: Embedding model name for token counting
        tokens_per_chunk: Maximum tokens per chunk
        min_chunk_length: Minimum chunk length in characters
        
    Returns:
        List of chunks
    """
    logger.info(f"Chunking {len(cleaned_documents)} documents")
    
    chunker = HybridSmartChunker(
        model_name=model_name,
        tokens_per_chunk=tokens_per_chunk,
        min_chunk_length=min_chunk_length,
    )
    
    all_chunks = []
    for doc in cleaned_documents:
        try:
            chunks = chunker.handle(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to chunk document {doc.id}: {e}")
    
    logger.success(f"Created {len(all_chunks)} chunks from {len(cleaned_documents)} documents")
    
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="chunks",
        metadata={
            "input_documents": len(cleaned_documents),
            "output_chunks": len(all_chunks),
            "avg_chunks_per_doc": len(all_chunks) / len(cleaned_documents) if cleaned_documents else 0,
            "model_name": model_name,
            "tokens_per_chunk": tokens_per_chunk,
        }
    )
    
    return all_chunks


@step
def embed_chunks(
    chunks: List[Chunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> Annotated[List[EmbeddedChunk], "embedded_chunks"]:
    """
    Generate embeddings for chunks.
    
    Args:
        chunks: List of chunks
        model_name: Embedding model name
        batch_size: Batch size for embedding generation
        
    Returns:
        List of embedded chunks
    """
    logger.info(f"Embedding {len(chunks)} chunks")
    
    embedder = SentenceTransformerEmbedder(
        model_name=model_name,
        use_singleton=True
    )
    
    embedded_chunks = embedder.handle(chunks)
    
    logger.success(f"Generated embeddings for {len(embedded_chunks)} chunks")
    
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_chunks",
        metadata={
            "total_chunks": len(chunks),
            "total_embedded": len(embedded_chunks),
            "model_name": model_name,
            "embedding_dim": len(embedded_chunks[0].embedding) if embedded_chunks else 0,
        }
    )
    
    return embedded_chunks


@step
def load_to_vector_db(
    embedded_chunks: List[EmbeddedChunk],
    collection_name: str = "hermes_documents",
    batch_size: int = 100,
) -> Annotated[str, "load_status"]:
    """
    Load embedded chunks to vector database.
    
    Args:
        embedded_chunks: List of embedded chunks
        collection_name: Vector DB collection name
        batch_size: Batch size for upload
        
    Returns:
        Status message
    """
    logger.info(f"Loading {len(embedded_chunks)} chunks to vector database")
    
    try:
        vector_store = VectorStoreClient(collection_name=collection_name)
        
        # Upload in batches
        total_uploaded = 0
        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i:i + batch_size]
            vector_store.add_chunks(batch)
            total_uploaded += len(batch)
            logger.debug(f"Uploaded batch {i//batch_size + 1}: {total_uploaded}/{len(embedded_chunks)}")
        
        logger.success(f"Loaded {total_uploaded} chunks to vector database")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="load_status",
            metadata={
                "total_chunks": len(embedded_chunks),
                "uploaded": total_uploaded,
                "collection": collection_name,
                "batch_size": batch_size,
            }
        )
        
        return f"Successfully loaded {total_uploaded} chunks"
        
    except Exception as e:
        logger.error(f"Failed to load to vector DB: {e}")
        return f"Failed: {str(e)}"


@step
def save_to_database(
    cleaned_documents: List[CleanedDocument],
) -> Annotated[str, "save_status"]:
    """
    Save cleaned documents to MongoDB.
    
    Args:
        cleaned_documents: List of cleaned documents
        
    Returns:
        Status message
    """
    logger.info(f"Saving {len(cleaned_documents)} documents to database")
    
    try:
        db_client = DatabaseClient()
        
        saved_count = 0
        for doc in cleaned_documents:
            try:
                db_client.save_cleaned_document(doc)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save document {doc.id}: {e}")
        
        logger.success(f"Saved {saved_count}/{len(cleaned_documents)} documents")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="save_status",
            metadata={
                "total_documents": len(cleaned_documents),
                "saved": saved_count,
                "failed": len(cleaned_documents) - saved_count,
            }
        )
        
        return f"Successfully saved {saved_count} documents"
        
    except Exception as e:
        logger.error(f"Failed to save to database: {e}")
        return f"Failed: {str(e)}"
