"""Package model for deployment."""

from pathlib import Path
from zenml import step
from loguru import logger

from hermes.deployment.packaging import ModelPackager


@step
def package_model_step(
    model_path: str | Path,
    output_dir: str | Path | None = None,
) -> str:
    """
    Package model as tar.gz for SageMaker deployment.
    
    Args:
        model_path: Path to trained model directory
        output_dir: Output directory for packaged model
        
    Returns:
        Path to packaged model tar.gz
    """
    logger.info(f"Packaging model from {model_path}")
    
    # Initialize packager
    packager = ModelPackager(
        model_path=model_path,
        output_dir=output_dir,
    )
    
    # Package model
    archive_path = packager.package_model()
    
    logger.info(f"Model packaged to: {archive_path}")
    
    return str(archive_path)