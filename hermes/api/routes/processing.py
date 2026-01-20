"""Processing endpoints."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from hermes.api.schemas import ProcessingRequest, ProcessingResponse
from hermes.pipelines.processing import processing_pipeline

router = APIRouter()


@router.post("/process", response_model=ProcessingResponse)
async def process_documents(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
) -> ProcessingResponse:
    """
    Process raw documents through cleaning, chunking, and embedding.
    
    Args:
        request: Processing configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Processing statistics
    """
    logger.info("Processing request received")
    
    try:
        # Run processing pipeline
        stats = processing_pipeline(
            author_id=request.author_id,
            limit=request.limit,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model
        )
        
        return ProcessingResponse(
            status="success",
            documents_processed=stats.get("total", 0),
            chunks_created=stats.get("total", 0),
            embeddings_generated=stats.get("total", 0),
            vectors_stored=stats.get("stored", 0)
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_processing_status():
    """Get processing pipeline status."""
    
    from hermes.core import CleanedDocument, Chunk, EmbeddedChunk
    
    try:
        cleaned_count = CleanedDocument.count()
        chunk_count = Chunk.count()
        embedded_count = EmbeddedChunk.count()
        
        return {
            "cleaned_documents": cleaned_count,
            "chunks": chunk_count,
            "embedded_chunks": embedded_count
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
