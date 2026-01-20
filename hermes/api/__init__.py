"""FastAPI service for Atlas LLM."""

from hermes.api.app import app, create_app
from hermes.api.schemas import (
    CollectionRequest,
    CollectionResponse,
    ProcessingRequest,
    ProcessingResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    HealthResponse
)

__all__ = [
    "app",
    "create_app",
    "CollectionRequest",
    "CollectionResponse",
    "ProcessingRequest",
    "ProcessingResponse",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "HealthResponse",
]
