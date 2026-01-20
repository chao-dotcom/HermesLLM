"""Upload model to S3."""

from pathlib import Path
from zenml import step
from loguru import logger

from hermes.deployment.packaging import S3ModelUploader


@step
def upload_to_s3_step(
    model_archive: str | Path,
    bucket_name: str,
    s3_prefix: str = "models",
    model_name: str | None = None,
    region: str = "us-east-1",
    aws_access_key: str | None = None,
    aws_secret_key: str | None = None,
) -> str:
    """
    Upload packaged model to S3.
    
    Args:
        model_archive: Path to model tar.gz
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix/folder
        model_name: Model name for S3 path
        region: AWS region
        aws_access_key: AWS access key (optional)
        aws_secret_key: AWS secret key (optional)
        
    Returns:
        S3 URI of uploaded model
    """
    logger.info(f"Uploading {model_archive} to S3 bucket {bucket_name}")
    
    # Initialize uploader
    uploader = S3ModelUploader(
        bucket_name=bucket_name,
        region=region,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
    )
    
    # Upload model
    s3_uri = uploader.upload_model(
        model_archive=model_archive,
        s3_prefix=s3_prefix,
        model_name=model_name,
    )
    
    logger.info(f"Model uploaded to: {s3_uri}")
    
    return s3_uri