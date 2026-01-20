"""Push trained model to HuggingFace Hub."""

import os
from pathlib import Path
from zenml import step
from loguru import logger

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not installed")


@step
def push_model_to_hub(
    model_path: str | Path,
    repo_id: str,
    commit_message: str = "Upload trained model",
    private: bool = False,
    token: str | None = None,
) -> str:
    """
    Push trained model to HuggingFace Hub.
    
    Args:
        model_path: Local path to trained model
        repo_id: HuggingFace repo ID (username/model-name)
        commit_message: Commit message
        private: Whether to make repo private
        token: HuggingFace API token (or use HUGGINGFACE_TOKEN env var)
        
    Returns:
        URL to the uploaded model
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")
    
    logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Get token
    token = token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required. Set HUGGINGFACE_TOKEN environment variable.")
    
    # Initialize API
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, token=token, private=private, exist_ok=True)
        logger.info(f"Repository created/verified: {repo_id}")
    except Exception as e:
        logger.warning(f"Could not create repo: {e}")
    
    # Upload folder
    url = api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
    )
    
    logger.info(f"Model successfully pushed to: {url}")
    
    return url
