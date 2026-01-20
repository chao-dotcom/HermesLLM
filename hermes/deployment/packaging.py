"""Model packaging and S3 upload utilities."""

import tarfile
import shutil
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. Install with: pip install boto3")


class ModelPackager:
    """Package models for SageMaker deployment."""
    
    def __init__(
        self,
        model_path: Path | str,
        output_dir: Optional[Path | str] = None,
    ) -> None:
        """
        Initialize model packager.
        
        Args:
            model_path: Path to trained model directory
            output_dir: Output directory for packaged model (default: model_path parent)
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir) if output_dir else self.model_path.parent
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def package_model(
        self,
        include_files: Optional[list[str]] = None,
        exclude_files: Optional[list[str]] = None,
    ) -> Path:
        """
        Package model as tar.gz for SageMaker.
        
        Args:
            include_files: List of file patterns to include (default: all)
            exclude_files: List of file patterns to exclude
            
        Returns:
            Path to packaged tar.gz file
        """
        logger.info(f"Packaging model from {self.model_path}")
        
        # Default includes: model weights, config, tokenizer
        if include_files is None:
            include_files = [
                "*.bin",
                "*.safetensors",
                "*.json",
                "*.txt",
                "*.model",
                "tokenizer.model",
                "tokenizer_config.json",
                "config.json",
                "generation_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ]
        
        # Default excludes: checkpoints, logs, cache
        if exclude_files is None:
            exclude_files = [
                "checkpoint-*",
                "*.log",
                "__pycache__",
                ".git",
                "*.pyc",
            ]
        
        # Create tar.gz
        archive_name = f"{self.model_path.name}.tar.gz"
        archive_path = self.output_dir / archive_name
        
        with tarfile.open(archive_path, "w:gz") as tar:
            for item in self.model_path.rglob("*"):
                if item.is_file():
                    # Check exclusions
                    should_exclude = any(
                        item.match(pattern) for pattern in exclude_files
                    )
                    if should_exclude:
                        continue
                    
                    # Check inclusions (if specified)
                    if include_files:
                        should_include = any(
                            item.match(pattern) for pattern in include_files
                        )
                        if not should_include:
                            continue
                    
                    # Add to archive
                    arcname = item.relative_to(self.model_path)
                    tar.add(item, arcname=arcname)
                    logger.debug(f"Added {arcname} to archive")
        
        logger.info(f"Model packaged to: {archive_path}")
        logger.info(f"Archive size: {archive_path.stat().st_size / (1024**2):.2f} MB")
        
        return archive_path
    
    def create_inference_code(
        self,
        code_dir: Path | str,
        entry_point: str = "inference.py",
    ) -> Path:
        """
        Create inference code directory for SageMaker.
        
        Args:
            code_dir: Directory containing inference code
            entry_point: Entry point script name
            
        Returns:
            Path to code tar.gz file
        """
        code_dir = Path(code_dir)
        
        if not code_dir.exists():
            raise FileNotFoundError(f"Code directory not found: {code_dir}")
        
        entry_point_path = code_dir / entry_point
        if not entry_point_path.exists():
            raise FileNotFoundError(f"Entry point not found: {entry_point_path}")
        
        logger.info(f"Packaging inference code from {code_dir}")
        
        # Create tar.gz
        archive_name = "code.tar.gz"
        archive_path = self.output_dir / archive_name
        
        with tarfile.open(archive_path, "w:gz") as tar:
            for item in code_dir.rglob("*"):
                if item.is_file() and not item.name.endswith(".pyc"):
                    arcname = item.relative_to(code_dir)
                    tar.add(item, arcname=arcname)
        
        logger.info(f"Inference code packaged to: {archive_path}")
        
        return archive_path


class S3ModelUploader:
    """Upload packaged models to S3."""
    
    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
    ) -> None:
        """
        Initialize S3 uploader.
        
        Args:
            bucket_name: S3 bucket name
            region: AWS region
            aws_access_key: AWS access key (optional, uses default credentials)
            aws_secret_key: AWS secret key (optional, uses default credentials)
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 required. Install with: pip install boto3")
        
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        client_kwargs = {"region_name": region}
        if aws_access_key and aws_secret_key:
            client_kwargs["aws_access_key_id"] = aws_access_key
            client_kwargs["aws_secret_access_key"] = aws_secret_key
        
        self.s3_client = boto3.client("s3", **client_kwargs)
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket exists: {self.bucket_name}")
        except ClientError:
            logger.info(f"Creating S3 bucket: {self.bucket_name}")
            try:
                if self.region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.region}
                    )
                logger.info(f"S3 bucket created: {self.bucket_name}")
            except ClientError as e:
                logger.error(f"Failed to create bucket: {e}")
                raise
    
    def upload_model(
        self,
        model_archive: Path | str,
        s3_prefix: str = "models",
        model_name: Optional[str] = None,
    ) -> str:
        """
        Upload model archive to S3.
        
        Args:
            model_archive: Path to model tar.gz file
            s3_prefix: S3 prefix/folder
            model_name: Custom model name (default: archive filename)
            
        Returns:
            S3 URI of uploaded model
        """
        model_archive = Path(model_archive)
        
        if not model_archive.exists():
            raise FileNotFoundError(f"Model archive not found: {model_archive}")
        
        # Determine S3 key
        if model_name:
            s3_key = f"{s3_prefix}/{model_name}/{model_archive.name}"
        else:
            s3_key = f"{s3_prefix}/{model_archive.name}"
        
        logger.info(f"Uploading {model_archive.name} to s3://{self.bucket_name}/{s3_key}")
        
        # Upload with progress
        file_size = model_archive.stat().st_size
        logger.info(f"File size: {file_size / (1024**2):.2f} MB")
        
        try:
            self.s3_client.upload_file(
                str(model_archive),
                self.bucket_name,
                s3_key,
            )
            
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Upload complete: {s3_uri}")
            
            return s3_uri
            
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            raise
    
    def download_model(
        self,
        s3_key: str,
        local_path: Path | str,
    ) -> Path:
        """
        Download model from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local path to save model
            
        Returns:
            Path to downloaded model
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")
        
        try:
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path),
            )
            
            logger.info(f"Download complete: {local_path}")
            return local_path
            
        except ClientError as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def list_models(self, prefix: str = "models") -> list[str]:
        """
        List models in S3 bucket.
        
        Args:
            prefix: S3 prefix to list
            
        Returns:
            List of S3 keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
            )
            
            if "Contents" not in response:
                return []
            
            return [obj["Key"] for obj in response["Contents"]]
            
        except ClientError as e:
            logger.error(f"Failed to list models: {e}")
            raise
