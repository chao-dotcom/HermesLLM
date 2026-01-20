"""Collection endpoints."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from hermes.api.schemas import CollectionRequest, CollectionResponse
from hermes.pipelines.collection import collection_pipeline

router = APIRouter()


@router.post("/collect", response_model=CollectionResponse)
async def collect_data(
    request: CollectionRequest,
    background_tasks: BackgroundTasks
) -> CollectionResponse:
    """
    Collect data from URLs.
    
    Args:
        request: Collection request with URLs and user info
        background_tasks: FastAPI background tasks
        
    Returns:
        Collection statistics
    """
    logger.info(f"Collection request for {len(request.urls)} URLs")
    
    try:
        # Run collection pipeline
        stats = collection_pipeline(
            user_id=request.user_id,
            full_name=request.full_name,
            links=request.urls
        )
        
        return CollectionResponse(
            status="success",
            total=stats["total"],
            success=stats["success"],
            failed=stats["failed"]
        )
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{user_id}")
async def get_collection_status(user_id: str):
    """Get collection status for user."""
    
    from hermes.core import ArticleDocument, PostDocument, RepositoryDocument
    
    try:
        articles = ArticleDocument.count(author_id=user_id)
        posts = PostDocument.count(author_id=user_id)
        repos = RepositoryDocument.count(author_id=user_id)
        
        return {
            "user_id": user_id,
            "articles": articles,
            "posts": posts,
            "repositories": repos,
            "total": articles + posts + repos
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
