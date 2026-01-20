"""RAG query endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from hermes.api.schemas import RAGQueryRequest, RAGQueryResponse
from hermes.rag import RAGPipeline

router = APIRouter()

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    Query the RAG system.
    
    Args:
        request: RAG query request
        
    Returns:
        Generated answer with sources
    """
    logger.info(f"RAG query: {request.query}")
    
    try:
        # Execute RAG query
        answer = rag_pipeline.query(
            query=request.query,
            use_query_expansion=request.use_query_expansion,
            system_prompt=request.system_prompt
        )
        
        return RAGQueryResponse(
            query=request.query,
            answer=answer,
            sources=[],  # TODO: Return source chunks
            metadata={
                "query_expansion": request.use_query_expansion,
                "top_k": request.top_k
            }
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_rag_stream(request: RAGQueryRequest):
    """
    Query RAG system with streaming response.
    
    Args:
        request: RAG query request
        
    Returns:
        Streaming response
    """
    logger.info(f"Streaming RAG query: {request.query}")
    
    try:
        from hermes.inference import StreamingPredictor
        
        # TODO: Implement full RAG with streaming
        # For now, just stream the LLM response
        predictor = StreamingPredictor()
        
        async def generate():
            for chunk in predictor.predict_stream(
                prompt=request.query,
                system_prompt=request.system_prompt
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Streaming query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
