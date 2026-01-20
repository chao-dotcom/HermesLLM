"""AWS SageMaker deployment and inference."""

from hermes.deployment.config import SageMakerConfig, InferenceConfig, DeploymentJobConfig
from hermes.deployment.deployer import SageMakerDeployer
from hermes.deployment.inference import SageMakerInferenceClient, SageMakerRAGClient
from hermes.deployment.packaging import ModelPackager, S3ModelUploader

try:
    import boto3
    import sagemaker
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False

__all__ = [
    "SageMakerConfig",
    "InferenceConfig",
    "DeploymentJobConfig",
    "SageMakerDeployer",
    "SageMakerInferenceClient",
    "SageMakerRAGClient",
    "ModelPackager",
    "S3ModelUploader",
    "SAGEMAKER_AVAILABLE",
]
