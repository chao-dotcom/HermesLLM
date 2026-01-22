"""
ML Inference Service

FastAPI-based inference server for HermesLLM models.
"""

from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from hermes.inference.predictor import ModelPredictor
from hermes.rag.pipeline import RAGPipeline


# Initialize FastAPI app
app = FastAPI(
    title="HermesLLM Inference API",
    description="Production-ready inference service for HermesLLM models",
    version="1.0.0",
)


# Request/Response Models
class InferenceRequest(BaseModel):
    """Request model for inference."""
    prompt: str = Field(..., description="Input prompt for the model")
    model_id: Optional[str] = Field(None, description="Model ID to use")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    system_prompt: Optional[str] = Field(None, description="System prompt")


class InferenceResponse(BaseModel):
    """Response model for inference."""
    generated_text: str = Field(..., description="Generated text")
    model_id: str = Field(..., description="Model used")
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")
    total_tokens: int = Field(..., description="Total tokens")


class RAGRequest(BaseModel):
    """Request model for RAG."""
    query: str = Field(..., description="User query")
    k: int = Field(5, description="Number of documents to retrieve")
    model_id: Optional[str] = Field(None, description="Model ID for generation")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")


class RAGResponse(BaseModel):
    """Response model for RAG."""
    answer: str = Field(..., description="Generated answer")
    retrieved_documents: List[Dict[str, Any]] = Field(..., description="Retrieved context")
    model_id: str = Field(..., description="Model used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: int


# Global state
_predictor: Optional[ModelPredictor] = None
_rag_pipeline: Optional[RAGPipeline] = None
_default_model_id: str = "meta-llama/Llama-2-7b-hf"


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global _predictor, _rag_pipeline
    
    logger.info("Initializing ML inference service...")
    
    try:
        # Initialize predictor
        _predictor = ModelPredictor(model_id=_default_model_id)
        logger.success(f"Loaded model: {_default_model_id}")
        
        # Initialize RAG pipeline
        _rag_pipeline = RAGPipeline()
        logger.success("Initialized RAG pipeline")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.warning("Service starting without models loaded")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ML inference service...")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "running",
        "version": "1.0.0",
        "models_loaded": 1 if _predictor else 0,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if _predictor else "unhealthy",
        "version": "1.0.0",
        "models_loaded": 1 if _predictor else 0,
    }


@app.post("/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Generate text from a prompt.
    
    Examples:
        ```
        curl -X POST http://localhost:8000/v1/inference \
            -H "Content-Type: application/json" \
            -d '{
                "prompt": "Explain machine learning in simple terms",
                "max_tokens": 200,
                "temperature": 0.7
            }'
        ```
    """
    if not _predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Inference request: {request.prompt[:50]}...")
        
        # Generate response
        result = _predictor.predict(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            system_prompt=request.system_prompt,
        )
        
        return InferenceResponse(
            generated_text=result["generated_text"],
            model_id=result.get("model_id", _default_model_id),
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    Answer query using RAG (Retrieval-Augmented Generation).
    
    Examples:
        ```
        curl -X POST http://localhost:8000/v1/rag \
            -H "Content-Type: application/json" \
            -d '{
                "query": "What are the benefits of RAG systems?",
                "k": 5,
                "max_tokens": 300
            }'
        ```
    """
    if not _rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not loaded")
    
    try:
        logger.info(f"RAG query: {request.query[:50]}...")
        
        # Run RAG pipeline
        result = _rag_pipeline.generate_answer(
            query=request.query,
            k=request.k,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        # Format retrieved documents
        retrieved_docs = [
            {
                "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "metadata": doc.metadata,
                "score": doc.score if hasattr(doc, "score") else None,
            }
            for doc in result.get("retrieved_documents", [])
        ]
        
        return RAGResponse(
            answer=result["answer"],
            retrieved_documents=retrieved_docs,
            model_id=result.get("model_id", _default_model_id),
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/reload-model")
async def reload_model(model_id: str):
    """
    Reload model with a different model ID.
    
    Examples:
        ```
        curl -X POST "http://localhost:8000/v1/reload-model?model_id=meta-llama/Llama-2-13b-hf"
        ```
    """
    global _predictor, _default_model_id
    
    try:
        logger.info(f"Reloading model: {model_id}")
        
        _predictor = ModelPredictor(model_id=model_id)
        _default_model_id = model_id
        
        logger.success(f"Model reloaded: {model_id}")
        
        return {"status": "success", "model_id": model_id}
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point for the service."""
    import uvicorn
    
    logger.info("Starting HermesLLM inference service...")
    uvicorn.run(
        "hermes.tools.ml_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
