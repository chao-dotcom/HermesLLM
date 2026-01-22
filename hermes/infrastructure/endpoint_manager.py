"""SageMaker endpoint management utilities.

This module provides utilities for managing SageMaker endpoints including
creation, deletion, and monitoring operations.
"""

from typing import Optional

from loguru import logger

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ModuleNotFoundError:
    BOTO3_AVAILABLE = False
    logger.warning(
        "boto3 not available. Install with: pip install boto3 botocore\n"
        "Or: poetry add boto3 botocore"
    )

from hermes.config import settings


class SageMakerEndpointManager:
    """Manager for SageMaker endpoint operations.

    Provides high-level interface for creating, deleting, and managing
    SageMaker inference endpoints.

    Attributes:
        region: AWS region for SageMaker operations
        client: boto3 SageMaker client

    Example:
        >>> manager = SageMakerEndpointManager()
        >>> manager.delete_endpoint("my-endpoint")
        >>> status = manager.get_endpoint_status("my-endpoint")
    """

    def __init__(
        self,
        region: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
    ):
        """Initialize SageMaker endpoint manager.

        Args:
            region: AWS region (default: from settings)
            aws_access_key: AWS access key (default: from settings)
            aws_secret_key: AWS secret key (default: from settings)

        Raises:
            RuntimeError: If boto3 not available
            ValueError: If AWS credentials not configured
        """
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 is not installed. Cannot create SageMaker client.")

        self.region = region or settings.AWS_REGION
        self.aws_access_key = aws_access_key or settings.AWS_ACCESS_KEY
        self.aws_secret_key = aws_secret_key or settings.AWS_SECRET_KEY

        if not self.region:
            raise ValueError("AWS_REGION is not set")
        if not self.aws_access_key:
            raise ValueError("AWS_ACCESS_KEY is not set")
        if not self.aws_secret_key:
            raise ValueError("AWS_SECRET_KEY is not set")

        self.client = boto3.client(
            "sagemaker",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
        )

    def get_endpoint_status(self, endpoint_name: str) -> Optional[str]:
        """Get the status of a SageMaker endpoint.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            str: Endpoint status (InService, Creating, Failed, etc.) or None if not found

        Example:
            >>> status = manager.get_endpoint_status("my-endpoint")
            >>> print(status)
            InService
        """
        try:
            response = self.client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]
            logger.info(f"Endpoint '{endpoint_name}' status: {status}")
            return status
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                logger.warning(f"Endpoint '{endpoint_name}' not found")
                return None
            else:
                logger.error(f"Failed to get endpoint status: {e}")
                raise

    def delete_endpoint(
        self, endpoint_name: str, delete_config: bool = True, delete_model: bool = True
    ) -> None:
        """Delete a SageMaker endpoint.

        Optionally also deletes the associated endpoint configuration and model.

        Args:
            endpoint_name: Name of the endpoint to delete
            delete_config: Whether to delete endpoint configuration (default: True)
            delete_model: Whether to delete model (default: True)

        Example:
            >>> manager.delete_endpoint("my-endpoint", delete_config=True)
        """
        config_name = None
        model_name = None

        # Get endpoint configuration name before deleting endpoint
        if delete_config or delete_model:
            try:
                response = self.client.describe_endpoint(EndpointName=endpoint_name)
                config_name = response.get("EndpointConfigName")
                logger.debug(f"Endpoint config: {config_name}")
            except ClientError as e:
                logger.warning(f"Failed to get endpoint configuration: {e}")

        # Get model name from endpoint configuration
        if delete_model and config_name:
            try:
                response = self.client.describe_endpoint_config(EndpointConfigName=config_name)
                variants = response.get("ProductionVariants", [])
                if variants:
                    model_name = variants[0].get("ModelName")
                    logger.debug(f"Model name: {model_name}")
            except ClientError as e:
                logger.warning(f"Failed to get model name: {e}")

        # Delete endpoint
        try:
            self.client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint '{endpoint_name}' deletion initiated")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                logger.warning(f"Endpoint '{endpoint_name}' not found")
            else:
                logger.error(f"Failed to delete endpoint: {e}")
                raise

        # Delete endpoint configuration
        if delete_config and config_name:
            try:
                self.client.delete_endpoint_config(EndpointConfigName=config_name)
                logger.info(f"Endpoint configuration '{config_name}' deleted")
            except ClientError as e:
                logger.warning(f"Failed to delete endpoint config: {e}")

        # Delete model
        if delete_model and model_name:
            try:
                self.client.delete_model(ModelName=model_name)
                logger.info(f"Model '{model_name}' deleted")
            except ClientError as e:
                logger.warning(f"Failed to delete model: {e}")

    def list_endpoints(
        self, name_contains: Optional[str] = None, status_equals: Optional[str] = None
    ) -> list[dict]:
        """List SageMaker endpoints.

        Args:
            name_contains: Filter by name substring (optional)
            status_equals: Filter by status (InService, Creating, etc.) (optional)

        Returns:
            list: List of endpoint dictionaries with EndpointName and EndpointStatus

        Example:
            >>> endpoints = manager.list_endpoints(status_equals="InService")
            >>> for ep in endpoints:
            ...     print(f"{ep['EndpointName']}: {ep['EndpointStatus']}")
        """
        try:
            params = {}
            if name_contains:
                params["NameContains"] = name_contains
            if status_equals:
                params["StatusEquals"] = status_equals

            response = self.client.list_endpoints(**params)
            endpoints = response.get("Endpoints", [])

            logger.info(f"Found {len(endpoints)} endpoints")
            return endpoints

        except ClientError as e:
            logger.error(f"Failed to list endpoints: {e}")
            return []

    def get_endpoint_metrics(self, endpoint_name: str) -> dict:
        """Get metrics for a SageMaker endpoint.

        Retrieves information about the endpoint including production variants,
        instance counts, and current status.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            dict: Endpoint metrics and configuration

        Example:
            >>> metrics = manager.get_endpoint_metrics("my-endpoint")
            >>> print(metrics["InstanceCount"])
        """
        try:
            response = self.client.describe_endpoint(EndpointName=endpoint_name)

            metrics = {
                "EndpointName": response["EndpointName"],
                "EndpointArn": response["EndpointArn"],
                "EndpointStatus": response["EndpointStatus"],
                "CreationTime": response["CreationTime"],
                "LastModifiedTime": response["LastModifiedTime"],
                "ProductionVariants": [],
            }

            for variant in response.get("ProductionVariants", []):
                metrics["ProductionVariants"].append(
                    {
                        "VariantName": variant["VariantName"],
                        "CurrentInstanceCount": variant.get("CurrentInstanceCount", 0),
                        "DesiredInstanceCount": variant.get("DesiredInstanceCount", 0),
                        "CurrentWeight": variant.get("CurrentWeight", 0.0),
                    }
                )

            logger.info(f"Retrieved metrics for endpoint '{endpoint_name}'")
            return metrics

        except ClientError as e:
            logger.error(f"Failed to get endpoint metrics: {e}")
            raise

    def wait_for_endpoint(
        self, endpoint_name: str, target_status: str = "InService", max_wait: int = 600
    ) -> bool:
        """Wait for endpoint to reach target status.

        Args:
            endpoint_name: Name of the endpoint
            target_status: Target status to wait for (default: InService)
            max_wait: Maximum wait time in seconds (default: 600)

        Returns:
            bool: True if target status reached, False if timeout

        Example:
            >>> success = manager.wait_for_endpoint("my-endpoint", "InService")
        """
        import time

        start_time = time.time()

        while time.time() - start_time < max_wait:
            status = self.get_endpoint_status(endpoint_name)

            if status == target_status:
                logger.info(f"Endpoint '{endpoint_name}' reached status: {target_status}")
                return True

            if status in ["Failed", "Deleting"]:
                logger.error(f"Endpoint '{endpoint_name}' in failed state: {status}")
                return False

            logger.debug(f"Waiting for endpoint... Current status: {status}")
            time.sleep(30)

        logger.warning(f"Timeout waiting for endpoint '{endpoint_name}'")
        return False


def delete_endpoint_and_config(
    endpoint_name: str,
    delete_config: bool = True,
    delete_model: bool = True,
    region: Optional[str] = None,
) -> None:
    """Delete a SageMaker endpoint and its configuration.

    Convenience function for deleting endpoints without creating a manager instance.

    Args:
        endpoint_name: Name of the endpoint to delete
        delete_config: Whether to delete endpoint configuration (default: True)
        delete_model: Whether to delete associated model (default: True)
        region: AWS region (default: from settings)

    Example:
        >>> delete_endpoint_and_config("my-old-endpoint")
    """
    try:
        manager = SageMakerEndpointManager(region=region)
        manager.delete_endpoint(
            endpoint_name=endpoint_name, delete_config=delete_config, delete_model=delete_model
        )
    except Exception as e:
        logger.error(f"Failed to delete endpoint: {e}")
        raise
