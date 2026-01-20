"""FastAPI application for Atlas LLM."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from hermes.config import get_settings
from hermes.storage.database import MongoDBConnection
from hermes.storage.vector_store import QdrantStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Atlas LLM API")
    
    # Initialize connections
    settings = get_settings()
    db = MongoDBConnection()
    vector_store = QdrantStore()
    
    logger.info("Connections initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Atlas LLM API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Atlas LLM API",
        description="API for Atlas LLM data collection, processing, and RAG",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routers
    from hermes.api.routes import collection, processing, rag, health
    
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(collection.router, prefix="/api/v1/collection", tags=["Collection"])
    app.include_router(processing.router, prefix="/api/v1/processing", tags=["Processing"])
    app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "hermes.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
