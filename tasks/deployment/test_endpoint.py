"""Test deployed SageMaker endpoint."""

from typing import Dict, Any
from zenml import step
from loguru import logger

from hermes.deployment import SageMakerInferenceClient, InferenceConfig


@step
def test_endpoint_step(
    endpoint_name: str,
    test_prompt: str = "What is machine learning?",
    region: str = "us-east-1",
    aws_access_key: str | None = None,
    aws_secret_key: str | None = None,
) -> Dict[str, Any]:
    """
    Test deployed SageMaker endpoint.
    
    Args:
        endpoint_name: SageMaker endpoint name
        test_prompt: Test prompt for inference
        region: AWS region
        aws_access_key: AWS access key (optional)
        aws_secret_key: AWS secret key (optional)
        
    Returns:
        Test results
    """
    logger.info(f"Testing endpoint: {endpoint_name}")
    
    # Initialize client
    client = SageMakerInferenceClient(
        endpoint_name=endpoint_name,
        region=region,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        config=InferenceConfig(max_new_tokens=100),
    )
    
    # Health check
    is_healthy = client.health_check()
    
    if not is_healthy:
        logger.error(f"Endpoint {endpoint_name} health check failed")
        return {
            "endpoint_name": endpoint_name,
            "status": "unhealthy",
            "test_passed": False,
        }
    
    # Test inference
    logger.info(f"Testing with prompt: {test_prompt}")
    
    try:
        response = client.generate(test_prompt, max_new_tokens=100)
        
        logger.info(f"Test inference successful")
        logger.info(f"Response: {response[:200]}...")  # Log first 200 chars
        
        return {
            "endpoint_name": endpoint_name,
            "status": "healthy",
            "test_passed": True,
            "test_prompt": test_prompt,
            "response": response,
        }
        
    except Exception as e:
        logger.error(f"Test inference failed: {e}")
        return {
            "endpoint_name": endpoint_name,
            "status": "error",
            "test_passed": False,
            "error": str(e),
        }