"""SageMaker deployment pipeline."""

from zenml import pipeline
from loguru import logger

from tasks.deployment import (
    package_model_step,
    upload_to_s3_step,
    deploy_endpoint_step,
    test_endpoint_step,
)


@pipeline
def sagemaker_deployment_pipeline(
    model_path: str,
    model_name: str,
    role_arn: str,
    s3_bucket: str,
    endpoint_name: str | None = None,
    instance_type: str = "ml.g5.2xlarge",
    instance_count: int = 1,
    region: str = "us-east-1",
    test_prompt: str = "What is machine learning?",
) -> None:
    """
    Complete SageMaker deployment pipeline.
    
    Steps:
    1. Package model as tar.gz
    2. Upload to S3
    3. Deploy to SageMaker endpoint
    4. Test endpoint
    
    Args:
        model_path: Local path to trained model
        model_name: Name for the model
        role_arn: AWS IAM role ARN for SageMaker
        s3_bucket: S3 bucket for model storage
        endpoint_name: SageMaker endpoint name (optional)
        instance_type: EC2 instance type
        instance_count: Number of instances
        region: AWS region
        test_prompt: Test prompt for validation
    """
    logger.info(f"Starting SageMaker deployment pipeline for {model_name}")
    
    # Step 1: Package model
    model_archive = package_model_step(model_path=model_path)
    
    # Step 2: Upload to S3
    s3_uri = upload_to_s3_step(
        model_archive=model_archive,
        bucket_name=s3_bucket,
        model_name=model_name,
        region=region,
    )
    
    # Step 3: Deploy endpoint
    deployed_endpoint = deploy_endpoint_step(
        model_data_s3_uri=s3_uri,
        model_name=model_name,
        role_arn=role_arn,
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        instance_count=instance_count,
        region=region,
        wait=True,
    )
    
    # Step 4: Test endpoint
    test_results = test_endpoint_step(
        endpoint_name=deployed_endpoint,
        test_prompt=test_prompt,
        region=region,
    )
    
    logger.info(f"Deployment pipeline completed!")
    logger.info(f"Endpoint: {deployed_endpoint}")
    logger.info(f"Test passed: {test_results.get('test_passed', False)}")


@pipeline
def sagemaker_update_pipeline(
    model_path: str,
    existing_endpoint: str,
    s3_bucket: str,
    region: str = "us-east-1",
) -> None:
    """
    Update existing SageMaker endpoint with new model.
    
    Args:
        model_path: Local path to new model
        existing_endpoint: Existing endpoint to update
        s3_bucket: S3 bucket for model storage
        region: AWS region
    """
    logger.info(f"Updating endpoint: {existing_endpoint}")
    
    # Package new model
    model_archive = package_model_step(model_path=model_path)
    
    # Upload to S3
    s3_uri = upload_to_s3_step(
        model_archive=model_archive,
        bucket_name=s3_bucket,
        model_name=f"{existing_endpoint}-update",
        region=region,
    )
    
    logger.info(f"New model uploaded: {s3_uri}")
    logger.info("Note: Use SageMakerDeployer.update_endpoint() to complete update")
