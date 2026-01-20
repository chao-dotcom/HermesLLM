"""
AWS SageMaker Deployment Examples
=================================

This example demonstrates how to:
- Package and upload models to S3
- Deploy models to SageMaker endpoints
- Invoke endpoints for inference
- Use ZenML pipelines for deployment
"""

import os
from pathlib import Path
from loguru import logger

from hermes.deployment import (
    SageMakerConfig,
    SageMakerDeployer,
    SageMakerInferenceClient,
    SageMakerRAGClient,
    ModelPackager,
    S3ModelUploader,
    InferenceConfig,
)
from workflows.pipelines.deployment import sagemaker_deployment_pipeline


def example_package_model():
    """Example: Package model for deployment."""
    
    logger.info("=== Packaging Model Example ===")
    
    # Initialize packager
    packager = ModelPackager(
        model_path="models/my-finetuned-model",
        output_dir="models/packaged",
    )
    
    # Package model
    archive_path = packager.package_model()
    
    logger.info(f"Model packaged to: {archive_path}")


def example_upload_to_s3():
    """Example: Upload model to S3."""
    
    logger.info("=== Upload to S3 Example ===")
    
    # Initialize uploader
    uploader = S3ModelUploader(
        bucket_name="my-sagemaker-models",
        region="us-east-1",
    )
    
    # Upload model
    s3_uri = uploader.upload_model(
        model_archive="models/packaged/my-model.tar.gz",
        s3_prefix="models",
        model_name="my-llm-model",
    )
    
    logger.info(f"Model uploaded to: {s3_uri}")


def example_deploy_endpoint():
    """Example: Deploy model to SageMaker endpoint."""
    
    logger.info("=== Deploy Endpoint Example ===")
    
    # Configure deployment
    config = SageMakerConfig(
        region="us-east-1",
        role_arn=os.getenv("AWS_SAGEMAKER_ROLE_ARN"),  # Your SageMaker IAM role
        model_path=Path("models/my-finetuned-model"),
        model_name="my-llm-model",
        instance_type="ml.g5.2xlarge",  # GPU instance
        instance_count=1,
        s3_bucket="my-sagemaker-models",
        endpoint_name="my-llm-endpoint",
    )
    
    # Initialize deployer
    deployer = SageMakerDeployer(config)
    
    # Deploy (packages, uploads, and creates endpoint)
    endpoint_name = deployer.deploy(wait=True)
    
    logger.info(f"Endpoint deployed: {endpoint_name}")


def example_deploy_from_s3():
    """Example: Deploy model already in S3."""
    
    logger.info("=== Deploy from S3 Example ===")
    
    config = SageMakerConfig(
        region="us-east-1",
        role_arn=os.getenv("AWS_SAGEMAKER_ROLE_ARN"),
        model_path=Path("."),  # Not used when deploying from S3
        model_name="my-llm-model",
        instance_type="ml.g5.xlarge",
        s3_bucket="my-sagemaker-models",
        endpoint_name="my-llm-endpoint-v2",
    )
    
    deployer = SageMakerDeployer(config)
    
    # Deploy without packaging (model already in S3)
    endpoint_name = deployer.deploy(package_model=False, wait=True)
    
    logger.info(f"Endpoint deployed: {endpoint_name}")


def example_inference():
    """Example: Invoke SageMaker endpoint for inference."""
    
    logger.info("=== Inference Example ===")
    
    # Initialize client
    client = SageMakerInferenceClient(
        endpoint_name="my-llm-endpoint",
        region="us-east-1",
        config=InferenceConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
        ),
    )
    
    # Generate text
    prompt = "Explain what is machine learning in simple terms."
    response = client.generate(prompt)
    
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")


def example_batch_inference():
    """Example: Batch inference."""
    
    logger.info("=== Batch Inference Example ===")
    
    client = SageMakerInferenceClient(
        endpoint_name="my-llm-endpoint",
        region="us-east-1",
    )
    
    # Multiple prompts
    prompts = [
        "What is Python?",
        "Explain neural networks.",
        "What is the difference between AI and ML?",
    ]
    
    # Batch generate
    responses = client.batch_generate(prompts, max_new_tokens=200)
    
    for prompt, response in zip(prompts, responses):
        logger.info(f"Q: {prompt}")
        logger.info(f"A: {response}\n")


def example_streaming_inference():
    """Example: Streaming inference."""
    
    logger.info("=== Streaming Inference Example ===")
    
    client = SageMakerInferenceClient(
        endpoint_name="my-llm-endpoint",
        region="us-east-1",
    )
    
    prompt = "Write a short story about AI:"
    
    logger.info(f"Prompt: {prompt}")
    logger.info("Response (streaming):")
    
    # Stream response
    for chunk in client.stream_generate(prompt, max_new_tokens=300):
        print(chunk, end="", flush=True)
    
    print()  # New line


def example_rag_inference():
    """Example: RAG inference with SageMaker."""
    
    logger.info("=== RAG Inference Example ===")
    
    # Initialize RAG client
    client = SageMakerRAGClient(
        endpoint_name="my-llm-endpoint",
        region="us-east-1",
        config=InferenceConfig(
            max_new_tokens=300,
            temperature=0.7,
        ),
    )
    
    # Query with context
    query = "What is LoRA?"
    context = [
        "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique.",
        "It works by adding trainable rank decomposition matrices to existing weights.",
        "LoRA significantly reduces the number of trainable parameters.",
    ]
    
    answer = client.query_with_context(
        query=query,
        context=context,
        system_prompt="You are a helpful AI assistant. Answer based on the context.",
    )
    
    logger.info(f"Query: {query}")
    logger.info(f"Answer: {answer}")


def example_zenml_pipeline():
    """Example: Deploy using ZenML pipeline."""
    
    logger.info("=== ZenML Deployment Pipeline Example ===")
    
    # Run deployment pipeline
    sagemaker_deployment_pipeline(
        model_path="models/my-finetuned-model",
        model_name="my-llm-model",
        role_arn=os.getenv("AWS_SAGEMAKER_ROLE_ARN"),
        s3_bucket="my-sagemaker-models",
        endpoint_name="my-llm-endpoint",
        instance_type="ml.g5.2xlarge",
        region="us-east-1",
        test_prompt="What is machine learning?",
    )
    
    logger.info("Deployment pipeline completed!")


def example_update_endpoint():
    """Example: Update existing endpoint."""
    
    logger.info("=== Update Endpoint Example ===")
    
    config = SageMakerConfig(
        region="us-east-1",
        role_arn=os.getenv("AWS_SAGEMAKER_ROLE_ARN"),
        model_path=Path("models/my-new-model"),
        model_name="my-llm-model-v2",
        instance_type="ml.g5.xlarge",
        s3_bucket="my-sagemaker-models",
    )
    
    deployer = SageMakerDeployer(config)
    
    # Package and upload new model
    new_model_s3 = deployer.package_and_upload_model()
    
    # Update endpoint
    deployer.update_endpoint(
        endpoint_name="my-llm-endpoint",
        new_model_data=new_model_s3,
    )
    
    logger.info("Endpoint updated!")


def example_endpoint_management():
    """Example: Manage endpoints."""
    
    logger.info("=== Endpoint Management Example ===")
    
    config = SageMakerConfig(
        region="us-east-1",
        role_arn=os.getenv("AWS_SAGEMAKER_ROLE_ARN"),
        model_path=Path("."),
        model_name="my-model",
    )
    
    deployer = SageMakerDeployer(config)
    
    # List all endpoints
    endpoints = deployer.list_endpoints()
    logger.info(f"Found {len(endpoints)} endpoints:")
    for ep in endpoints:
        logger.info(f"  - {ep['EndpointName']}: {ep['EndpointStatus']}")
    
    # Get endpoint status
    status = deployer.get_endpoint_status("my-llm-endpoint")
    logger.info(f"Endpoint status: {status}")
    
    # Delete endpoint
    # deployer.delete_endpoint("my-llm-endpoint", delete_model=True)


def example_inference_with_chat():
    """Example: Chat completion."""
    
    logger.info("=== Chat Completion Example ===")
    
    client = SageMakerInferenceClient(
        endpoint_name="my-llm-endpoint",
        region="us-east-1",
    )
    
    # Chat messages
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."},
        {"role": "user", "content": "What are its main features?"},
    ]
    
    response = client.chat(messages, max_new_tokens=200)
    
    logger.info(f"Chat response: {response}")


def example_health_check():
    """Example: Health check endpoint."""
    
    logger.info("=== Health Check Example ===")
    
    client = SageMakerInferenceClient(
        endpoint_name="my-llm-endpoint",
        region="us-east-1",
    )
    
    # Check health
    is_healthy = client.health_check()
    
    if is_healthy:
        logger.info("✓ Endpoint is healthy")
    else:
        logger.error("✗ Endpoint is unhealthy")
    
    # Get metrics
    metrics = client.get_endpoint_metrics()
    logger.info(f"Endpoint metrics: {metrics}")


if __name__ == "__main__":
    # Set AWS credentials
    # os.environ["AWS_ACCESS_KEY_ID"] = "your-key"
    # os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret"
    # os.environ["AWS_SAGEMAKER_ROLE_ARN"] = "arn:aws:iam::123456789012:role/SageMakerRole"
    
    # Choose which example to run
    
    # Deployment examples
    # example_package_model()
    # example_upload_to_s3()
    # example_deploy_endpoint()
    # example_deploy_from_s3()
    # example_zenml_pipeline()
    
    # Inference examples
    # example_inference()
    # example_batch_inference()
    # example_streaming_inference()
    # example_rag_inference()
    # example_inference_with_chat()
    
    # Management examples
    # example_update_endpoint()
    # example_endpoint_management()
    # example_health_check()
    
    logger.info("Uncomment one of the examples above to run it!")
