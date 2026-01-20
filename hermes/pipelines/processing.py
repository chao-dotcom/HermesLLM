"""Data processing pipeline using ZenML."""

from typing import List, Dict, Any

from zenml import pipeline, step
from loguru import logger

from hermes.core import (
    ArticleDocument,
    PostDocument,
    RepositoryDocument,
    CleanedDocument,
    Chunk,
    EmbeddedChunk
)
from hermes.processing import (
    CleaningDispatcher,
    TextChunker,
    CodeChunker,
    SentenceTransformerEmbedder
)
from hermes.storage.vector_store import QdrantStore


@step
def load_raw_documents_step(
    author_id: str | None = None,
    limit: int | None = None
) -> List[Any]:
    """
    Load raw documents from MongoDB.
    
    Args:
        author_id: Filter by author ID
        limit: Maximum documents to load
        
    Returns:
        List of raw documents
    """
    logger.info(f"Loading raw documents (author_id={author_id}, limit={limit})")
    
    documents = []
    
    # Load articles
    articles = ArticleDocument.find(author_id=author_id, limit=limit)
    documents.extend(articles)
    
    # Load posts
    posts = PostDocument.find(author_id=author_id, limit=limit)
    documents.extend(posts)
    
    # Load repositories
    repos = RepositoryDocument.find(author_id=author_id, limit=limit)
    documents.extend(repos)
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


@step
def clean_documents_step(
    raw_documents: List[Any]
) -> List[CleanedDocument]:
    """
    Clean raw documents.
    
    Args:
        raw_documents: List of raw documents
        
    Returns:
        List of cleaned documents
    """
    logger.info(f"Cleaning {len(raw_documents)} documents")
    
    dispatcher = (
        CleaningDispatcher.build()
        .register_article()
        .register_post()
        .register_repository()
    )
    
    cleaned = []
    for doc in raw_documents:
        try:
            cleaner = dispatcher.get_cleaner(doc)
            cleaned_doc = cleaner.handle(doc)
            cleaned_doc.save()
            cleaned.append(cleaned_doc)
        except Exception as e:
            logger.error(f"Failed to clean document {doc.id}: {e}")
    
    logger.info(f"Cleaned {len(cleaned)} documents")
    return cleaned


@step
def chunk_documents_step(
    cleaned_documents: List[CleanedDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Chunk]:
    """
    Chunk cleaned documents.
    
    Args:
        cleaned_documents: List of cleaned documents
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunks
    """
    logger.info(f"Chunking {len(cleaned_documents)} documents")
    
    text_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    code_chunker = CodeChunker(chunk_size=chunk_size * 2, chunk_overlap=chunk_overlap * 2)
    
    all_chunks = []
    for doc in cleaned_documents:
        try:
            # Use code chunker for repositories, text chunker for others
            if doc.type == "repository":
                chunks = code_chunker.handle(doc)
            else:
                chunks = text_chunker.handle(doc)
            
            # Save chunks
            for chunk in chunks:
                chunk.save()
            
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to chunk document {doc.id}: {e}")
    
    logger.info(f"Created {len(all_chunks)} chunks")
    return all_chunks


@step
def embed_chunks_step(
    chunks: List[Chunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[EmbeddedChunk]:
    """
    Generate embeddings for chunks.
    
    Args:
        chunks: List of chunks
        model_name: Embedding model name
        
    Returns:
        List of embedded chunks
    """
    logger.info(f"Embedding {len(chunks)} chunks")
    
    embedder = SentenceTransformerEmbedder(model_name=model_name)
    
    try:
        embedded_chunks = embedder.handle(chunks)
        
        # Save to MongoDB
        for chunk in embedded_chunks:
            chunk.save()
        
        logger.info(f"Generated {len(embedded_chunks)} embeddings")
        return embedded_chunks
    except Exception as e:
        logger.error(f"Failed to embed chunks: {e}")
        return []


@step
def store_vectors_step(
    embedded_chunks: List[EmbeddedChunk],
    collection_name: str = "chunks"
) -> Dict[str, int]:
    """
    Store embeddings in Qdrant.
    
    Args:
        embedded_chunks: List of embedded chunks
        collection_name: Qdrant collection name
        
    Returns:
        Storage statistics
    """
    logger.info(f"Storing {len(embedded_chunks)} vectors in Qdrant")
    
    vector_store = QdrantStore()
    
    # Ensure collection exists
    if embedded_chunks:
        vector_store.create_collection(
            collection_name=collection_name,
            vector_size=len(embedded_chunks[0].embedding)
        )
    
    # Store vectors
    success_count = 0
    for chunk in embedded_chunks:
        try:
            vector_store.upsert(
                collection_name=collection_name,
                point_id=str(chunk.id),
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "document_id": str(chunk.document_id),
                    "author_id": str(chunk.author_id),
                    "platform": chunk.platform,
                    "index": chunk.index,
                    "metadata": chunk.metadata
                }
            )
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk.id}: {e}")
    
    stats = {
        "total": len(embedded_chunks),
        "stored": success_count,
        "failed": len(embedded_chunks) - success_count
    }
    
    logger.info(f"Stored {success_count}/{len(embedded_chunks)} vectors")
    return stats


@pipeline
def processing_pipeline(
    author_id: str | None = None,
    limit: int | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    collection_name: str = "chunks"
) -> Dict[str, Any]:
    """
    Complete data processing pipeline.
    
    Args:
        author_id: Filter by author ID
        limit: Maximum documents to process
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        embedding_model: Embedding model name
        collection_name: Qdrant collection name
        
    Returns:
        Processing statistics
    """
    # Step 1: Load raw documents
    raw_docs = load_raw_documents_step(author_id=author_id, limit=limit)
    
    # Step 2: Clean documents
    cleaned_docs = clean_documents_step(raw_documents=raw_docs)
    
    # Step 3: Chunk documents
    chunks = chunk_documents_step(
        cleaned_documents=cleaned_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Step 4: Embed chunks
    embedded = embed_chunks_step(chunks=chunks, model_name=embedding_model)
    
    # Step 5: Store in vector database
    stats = store_vectors_step(embedded_chunks=embedded, collection_name=collection_name)
    
    return stats
