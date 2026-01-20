"""Configuration for SageMaker deployment and inference."""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class SageMakerConfig(BaseModel):
    """Configuration for SageMaker deployment."""
    
    # AWS Configuration
    region: str = Field(default="us-east-1", description="AWS region")
    role_arn: str = Field(..., description="AWS IAM role ARN for SageMaker")
    aws_access_key: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_key: Optional[str] = Field(default=None, description="AWS secret key")
    
    # Model Configuration
    model_path: Path = Field(..., description="Local path to model")
    model_name: str = Field(..., description="Name for the SageMaker model")
    entry_point: str = Field(default="inference.py", description="Inference entry point script")
    source_dir: Optional[Path] = Field(default=None, description="Directory containing inference code")
    
    # Instance Configuration
    instance_type: str = Field(default="ml.g5.2xlarge", description="Instance type for deployment")
    instance_count: int = Field(default=1, description="Number of instances")
    
    # Framework Configuration
    framework_version: str = Field(default="2.1.0", description="PyTorch version")
    py_version: str = Field(default="py310", description="Python version")
    transformers_version: str = Field(default="4.40.0", description="Transformers version")
    
    # Endpoint Configuration
    endpoint_name: Optional[str] = Field(default=None, description="SageMaker endpoint name")
    inference_component_name: Optional[str] = Field(default=None, description="Inference component name")
    
    # S3 Configuration
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for model artifacts")
    s3_prefix: str = Field(default="models", description="S3 prefix for model artifacts")
    
    # HuggingFace Configuration
    huggingface_token: Optional[str] = Field(default=None, description="HuggingFace token")
    
    # Environment Variables
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    class Config:
        arbitrary_types_allowed = True


class InferenceConfig(BaseModel):
    """Configuration for SageMaker inference."""
    
    # Generation Parameters
    max_new_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling parameter")
    top_k: int = Field(default=50, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    return_full_text: bool = Field(default=False, description="Return full text including prompt")
    
    # Inference Configuration
    batch_size: int = Field(default=1, description="Batch size for inference")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    
    def to_payload_params(self) -> Dict[str, Any]:
        """Convert to SageMaker payload parameters."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "return_full_text": self.return_full_text,
        }


class DeploymentJobConfig(BaseModel):
    """Configuration for SageMaker training/fine-tuning jobs."""
    
    # Job Configuration
    job_name: Optional[str] = Field(default=None, description="Training job name")
    instance_type: str = Field(default="ml.g5.2xlarge", description="Instance type")
    instance_count: int = Field(default=1, description="Number of instances")
    volume_size: int = Field(default=100, description="EBS volume size in GB")
    max_runtime: int = Field(default=86400, description="Max runtime in seconds")
    
    # Framework
    framework: str = Field(default="huggingface", description="ML framework")
    pytorch_version: str = Field(default="2.1.0", description="PyTorch version")
    transformers_version: str = Field(default="4.40.0", description="Transformers version")
    py_version: str = Field(default="py310", description="Python version")
    
    # Training Configuration
    entry_point: str = Field(default="train.py", description="Training script")
    source_dir: Path = Field(..., description="Directory containing training code")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Training hyperparameters")
    
    # Environment
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    class Config:
        arbitrary_types_allowed = True
