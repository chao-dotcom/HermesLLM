"""Health check endpoints."""

from fastapi import APIRouter
from loguru import logger

from hermes.api.schemas import HealthResponse
from hermes.storage.database import MongoDBConnection
from hermes.storage.vector_store import QdrantStore

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and service status."""
    
    logger.info("Health check requested")
    
    services = {}
    
    # Check MongoDB
    try:
        db = MongoDBConnection()
        db.client.admin.command('ping')
        services["mongodb"] = "healthy"
    except Exception as e:
        logger.error(f"MongoDB unhealthy: {e}")
        services["mongodb"] = "unhealthy"
    
    # Check Qdrant
    try:
        vector_store = QdrantStore()
        vector_store.client.get_collections()
        services["qdrant"] = "healthy"
    except Exception as e:
        logger.error(f"Qdrant unhealthy: {e}")
        services["qdrant"] = "unhealthy"
    
    overall_status = "healthy" if all(
        status == "healthy" for status in services.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        services=services
    )
