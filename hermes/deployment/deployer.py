"""SageMaker model deployment and endpoint management."""

import time
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

try:
    import boto3
    from sagemaker.huggingface import HuggingFaceModel
    from sagemaker.session import Session
    from sagemaker import get_execution_role
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False
    logger.warning("SageMaker not installed. Install with: pip install sagemaker boto3")

from hermes.deployment.config import SageMakerConfig
from hermes.deployment.packaging import ModelPackager, S3ModelUploader


class SageMakerDeployer:
    """Deploy models to AWS SageMaker endpoints."""
    
    def __init__(self, config: SageMakerConfig) -> None:
        """
        Initialize SageMaker deployer.
        
        Args:
            config: SageMaker configuration
        """
        if not SAGEMAKER_AVAILABLE:
            raise ImportError("SageMaker required. Install with: pip install sagemaker boto3")
        
        self.config = config
        
        # Initialize AWS clients
        client_kwargs = {"region_name": config.region}
        if config.aws_access_key and config.aws_secret_key:
            client_kwargs["aws_access_key_id"] = config.aws_access_key
            client_kwargs["aws_secret_access_key"] = config.aws_secret_key
        
        self.sagemaker_client = boto3.client("sagemaker", **client_kwargs)
        self.sagemaker_runtime = boto3.client("sagemaker-runtime", **client_kwargs)
        
        # Initialize SageMaker session
        self.session = Session(
            boto_session=boto3.Session(**client_kwargs),
            sagemaker_client=self.sagemaker_client,
            sagemaker_runtime_client=self.sagemaker_runtime,
        )
        
        logger.info(f"Initialized SageMaker deployer for region: {config.region}")
    
    def package_and_upload_model(
        self,
        packager: Optional[ModelPackager] = None,
    ) -> str:
        """
        Package model and upload to S3.
        
        Args:
            packager: Custom model packager (creates default if None)
            
        Returns:
            S3 URI of uploaded model
        """
        # Create packager if not provided
        if packager is None:
            packager = ModelPackager(self.config.model_path)
        
        # Package model
        model_archive = packager.package_model()
        
        # Determine S3 bucket
        if self.config.s3_bucket:
            bucket_name = self.config.s3_bucket
        else:
            # Use SageMaker default bucket
            bucket_name = self.session.default_bucket()
            logger.info(f"Using SageMaker default bucket: {bucket_name}")
        
        # Upload to S3
        uploader = S3ModelUploader(
            bucket_name=bucket_name,
            region=self.config.region,
            aws_access_key=self.config.aws_access_key,
            aws_secret_key=self.config.aws_secret_key,
        )
        
        s3_uri = uploader.upload_model(
            model_archive=model_archive,
            s3_prefix=self.config.s3_prefix,
            model_name=self.config.model_name,
        )
        
        return s3_uri
    
    def create_model(
        self,
        model_data: str,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Create SageMaker model from S3 artifact.
        
        Args:
            model_data: S3 URI of model tar.gz
            model_name: Model name (uses config if None)
            
        Returns:
            SageMaker model name
        """
        model_name = model_name or self.config.model_name
        
        logger.info(f"Creating SageMaker model: {model_name}")
        
        # Set up environment variables
        environment = {
            "HF_MODEL_ID": model_name,
            "HF_TASK": "text-generation",
            **self.config.environment,
        }
        
        if self.config.huggingface_token:
            environment["HUGGING_FACE_HUB_TOKEN"] = self.config.huggingface_token
        
        # Create HuggingFace model
        huggingface_model = HuggingFaceModel(
            model_data=model_data,
            role=self.config.role_arn,
            transformers_version=self.config.transformers_version,
            pytorch_version=self.config.framework_version,
            py_version=self.config.py_version,
            env=environment,
            sagemaker_session=self.session,
            name=model_name,
        )
        
        logger.info(f"Model created: {model_name}")
        
        return huggingface_model
    
    def deploy_endpoint(
        self,
        model: Any,
        endpoint_name: Optional[str] = None,
        wait: bool = True,
    ) -> str:
        """
        Deploy model to SageMaker endpoint.
        
        Args:
            model: HuggingFaceModel instance
            endpoint_name: Endpoint name (generates from config if None)
            wait: Wait for endpoint to be in service
            
        Returns:
            Endpoint name
        """
        endpoint_name = endpoint_name or self.config.endpoint_name or f"{self.config.model_name}-endpoint"
        
        logger.info(f"Deploying endpoint: {endpoint_name}")
        logger.info(f"Instance type: {self.config.instance_type}")
        logger.info(f"Instance count: {self.config.instance_count}")
        
        # Deploy endpoint
        predictor = model.deploy(
            initial_instance_count=self.config.instance_count,
            instance_type=self.config.instance_type,
            endpoint_name=endpoint_name,
            wait=wait,
        )
        
        if wait:
            logger.info(f"Endpoint deployed and in service: {endpoint_name}")
        else:
            logger.info(f"Endpoint deployment initiated: {endpoint_name}")
        
        return endpoint_name
    
    def deploy(
        self,
        package_model: bool = True,
        wait: bool = True,
    ) -> str:
        """
        Complete deployment workflow.
        
        Args:
            package_model: Whether to package and upload model (False if already in S3)
            wait: Wait for endpoint to be in service
            
        Returns:
            Endpoint name
        """
        logger.info("Starting deployment workflow...")
        
        # Step 1: Package and upload model
        if package_model:
            logger.info("Step 1/3: Packaging and uploading model...")
            model_data = self.package_and_upload_model()
        else:
            # Assume model already in S3
            model_data = f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}/{self.config.model_name}/model.tar.gz"
            logger.info(f"Using existing model: {model_data}")
        
        # Step 2: Create SageMaker model
        logger.info("Step 2/3: Creating SageMaker model...")
        model = self.create_model(model_data)
        
        # Step 3: Deploy endpoint
        logger.info("Step 3/3: Deploying endpoint...")
        endpoint_name = self.deploy_endpoint(model, wait=wait)
        
        logger.info(f"✓ Deployment complete!")
        logger.info(f"Endpoint: {endpoint_name}")
        
        return endpoint_name
    
    def update_endpoint(
        self,
        endpoint_name: str,
        new_model_data: str,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None,
    ) -> None:
        """
        Update existing endpoint with new model or configuration.
        
        Args:
            endpoint_name: Existing endpoint name
            new_model_data: S3 URI of new model
            instance_type: New instance type (optional)
            instance_count: New instance count (optional)
        """
        logger.info(f"Updating endpoint: {endpoint_name}")
        
        # Create new model version
        model_name = f"{endpoint_name}-{int(time.time())}"
        model = self.create_model(new_model_data, model_name=model_name)
        
        # Update endpoint
        instance_type = instance_type or self.config.instance_type
        instance_count = instance_count or self.config.instance_count
        
        logger.info("Updating endpoint configuration...")
        
        try:
            # This will trigger an update
            model.deploy(
                initial_instance_count=instance_count,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
                update_endpoint=True,
            )
            
            logger.info(f"Endpoint updated: {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to update endpoint: {e}")
            raise
    
    def delete_endpoint(
        self,
        endpoint_name: str,
        delete_model: bool = True,
        delete_endpoint_config: bool = True,
    ) -> None:
        """
        Delete SageMaker endpoint.
        
        Args:
            endpoint_name: Endpoint to delete
            delete_model: Also delete associated model
            delete_endpoint_config: Also delete endpoint configuration
        """
        logger.info(f"Deleting endpoint: {endpoint_name}")
        
        try:
            # Get endpoint details before deletion
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            config_name = response["EndpointConfigName"]
            
            # Get config to find model
            config_response = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=config_name
            )
            model_name = config_response["ProductionVariants"][0]["ModelName"]
            
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint deleted: {endpoint_name}")
            
            # Delete endpoint config
            if delete_endpoint_config:
                self.sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
                logger.info(f"Endpoint config deleted: {config_name}")
            
            # Delete model
            if delete_model:
                self.sagemaker_client.delete_model(ModelName=model_name)
                logger.info(f"Model deleted: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            raise
    
    def list_endpoints(self) -> list[Dict[str, Any]]:
        """
        List all SageMaker endpoints.
        
        Returns:
            List of endpoint information
        """
        try:
            response = self.sagemaker_client.list_endpoints()
            endpoints = response.get("Endpoints", [])
            
            logger.info(f"Found {len(endpoints)} endpoints")
            
            return endpoints
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            raise
    
    def get_endpoint_status(self, endpoint_name: str) -> str:
        """
        Get endpoint status.
        
        Args:
            endpoint_name: Endpoint name
            
        Returns:
            Endpoint status (InService, Creating, Updating, Failed, etc.)
        """
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]
            
            logger.info(f"Endpoint {endpoint_name} status: {status}")
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get endpoint status: {e}")
            raise
