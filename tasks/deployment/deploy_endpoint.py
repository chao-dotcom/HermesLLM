"""Deploy SageMaker endpoint."""

from pathlib import Path
from zenml import step
from loguru import logger

from hermes.deployment import SageMakerConfig, SageMakerDeployer


@step
def deploy_endpoint_step(
    model_data_s3_uri: str,
    model_name: str,
    role_arn: str,
    endpoint_name: str | None = None,
    instance_type: str = "ml.g5.2xlarge",
    instance_count: int = 1,
    region: str = "us-east-1",
    aws_access_key: str | None = None,
    aws_secret_key: str | None = None,
    wait: bool = True,
) -> str:
    """
    Deploy model to SageMaker endpoint.
    
    Args:
        model_data_s3_uri: S3 URI of model tar.gz
        model_name: Model name
        role_arn: AWS IAM role ARN
        endpoint_name: Endpoint name (optional)
        instance_type: EC2 instance type
        instance_count: Number of instances
        region: AWS region
        aws_access_key: AWS access key (optional)
        aws_secret_key: AWS secret key (optional)
        wait: Wait for endpoint to be in service
        
    Returns:
        Endpoint name
    """
    logger.info(f"Deploying SageMaker endpoint for model: {model_name}")
    
    # Create config
    config = SageMakerConfig(
        region=region,
        role_arn=role_arn,
        model_path=Path("."),  # Not used since model already in S3
        model_name=model_name,
        instance_type=instance_type,
        instance_count=instance_count,
        endpoint_name=endpoint_name,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
    )
    
    # Initialize deployer
    deployer = SageMakerDeployer(config)
    
    # Create model
    model = deployer.create_model(model_data=model_data_s3_uri)
    
    # Deploy endpoint
    endpoint_name = deployer.deploy_endpoint(model=model, wait=wait)
    
    logger.info(f"Endpoint deployed: {endpoint_name}")
    
    return endpoint_name