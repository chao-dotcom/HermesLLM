"""SageMaker endpoint autoscaling utilities.

This module provides utilities for configuring auto-scaling policies for
SageMaker inference endpoints based on CloudWatch metrics.
"""

from abc import ABC, abstractmethod
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


class ScalingPolicy(ABC):
    """Abstract base class for scaling policies.

    Defines the interface for implementing different auto-scaling policies.
    """

    @abstractmethod
    def apply(self, client, resource_id: str, service_namespace: str, scalable_dimension: str):
        """Apply the scaling policy.

        Args:
            client: boto3 Application Auto Scaling client
            resource_id: Resource identifier (e.g., endpoint/my-endpoint)
            service_namespace: AWS service namespace (e.g., sagemaker)
            scalable_dimension: Dimension to scale (e.g., variant:DesiredInstanceCount)
        """
        pass


class TargetTrackingPolicy(ScalingPolicy):
    """Target tracking scaling policy.

    Automatically adjusts capacity to maintain a target value for a specified metric.
    This is the recommended scaling policy type for most use cases.

    Attributes:
        policy_name: Name for the scaling policy
        target_value: Target value for the metric
        predefined_metric_type: CloudWatch metric to track
        scale_in_cooldown: Cooldown period after scale in (seconds)
        scale_out_cooldown: Cooldown period after scale out (seconds)

    Example:
        >>> policy = TargetTrackingPolicy(
        ...     policy_name="my-policy",
        ...     target_value=70.0,
        ...     predefined_metric_type="SageMakerVariantInvocationsPerInstance"
        ... )
    """

    def __init__(
        self,
        policy_name: str,
        target_value: float,
        predefined_metric_type: str = "SageMakerVariantInvocationsPerInstance",
        scale_in_cooldown: int = 300,
        scale_out_cooldown: int = 60,
    ):
        """Initialize target tracking policy.

        Args:
            policy_name: Name for the scaling policy
            target_value: Target value to maintain (e.g., 70.0 for 70 invocations/instance)
            predefined_metric_type: Predefined metric type (default: SageMakerVariantInvocationsPerInstance)
                Options:
                - SageMakerVariantInvocationsPerInstance: Invocations per instance
                - SageMakerVariantInvocationsPerCopy: Invocations per inference component copy
            scale_in_cooldown: Seconds to wait after scale in before another scale in (default: 300)
            scale_out_cooldown: Seconds to wait after scale out before another scale out (default: 60)
        """
        self.policy_name = policy_name
        self.target_value = target_value
        self.predefined_metric_type = predefined_metric_type
        self.scale_in_cooldown = scale_in_cooldown
        self.scale_out_cooldown = scale_out_cooldown

    def apply(
        self,
        client,
        resource_id: str,
        service_namespace: str = "sagemaker",
        scalable_dimension: str = "sagemaker:variant:DesiredInstanceCount",
    ):
        """Apply the target tracking scaling policy.

        Args:
            client: boto3 Application Auto Scaling client
            resource_id: Resource ID (e.g., endpoint/my-endpoint/variant/AllTraffic)
            service_namespace: Service namespace (default: sagemaker)
            scalable_dimension: Scalable dimension (default: sagemaker:variant:DesiredInstanceCount)

        Raises:
            ClientError: If policy creation fails
        """
        try:
            response = client.put_scaling_policy(
                PolicyName=self.policy_name,
                PolicyType="TargetTrackingScaling",
                ServiceNamespace=service_namespace,
                ResourceId=resource_id,
                ScalableDimension=scalable_dimension,
                TargetTrackingScalingPolicyConfiguration={
                    "PredefinedMetricSpecification": {
                        "PredefinedMetricType": self.predefined_metric_type,
                    },
                    "TargetValue": self.target_value,
                    "ScaleInCooldown": self.scale_in_cooldown,
                    "ScaleOutCooldown": self.scale_out_cooldown,
                },
            )

            logger.info(f"Applied target tracking policy '{self.policy_name}'")
            logger.debug(f"Policy ARN: {response.get('PolicyARN')}")

        except ClientError as e:
            logger.error(f"Failed to apply scaling policy: {e}")
            raise


class StepScalingPolicy(ScalingPolicy):
    """Step scaling policy.

    Scales capacity in steps based on CloudWatch alarm thresholds.
    Provides more granular control than target tracking but requires more configuration.

    Attributes:
        policy_name: Name for the scaling policy
        adjustment_type: Type of adjustment (PercentChangeInCapacity, ChangeInCapacity, ExactCapacity)
        metric_aggregation_type: How to aggregate metric data
        step_adjustments: List of step adjustment configurations

    Example:
        >>> policy = StepScalingPolicy(
        ...     policy_name="step-policy",
        ...     step_adjustments=[
        ...         {"MetricIntervalLowerBound": 0, "ScalingAdjustment": 1},
        ...         {"MetricIntervalLowerBound": 10, "ScalingAdjustment": 2}
        ...     ]
        ... )
    """

    def __init__(
        self,
        policy_name: str,
        step_adjustments: list[dict],
        adjustment_type: str = "ChangeInCapacity",
        metric_aggregation_type: str = "Average",
        cooldown: int = 300,
    ):
        """Initialize step scaling policy.

        Args:
            policy_name: Name for the scaling policy
            step_adjustments: List of step adjustment dicts with MetricIntervalLowerBound
                and ScalingAdjustment
            adjustment_type: How to adjust capacity (default: ChangeInCapacity)
            metric_aggregation_type: How to aggregate metrics (default: Average)
            cooldown: Seconds to wait between scaling activities (default: 300)
        """
        self.policy_name = policy_name
        self.step_adjustments = step_adjustments
        self.adjustment_type = adjustment_type
        self.metric_aggregation_type = metric_aggregation_type
        self.cooldown = cooldown

    def apply(
        self,
        client,
        resource_id: str,
        service_namespace: str = "sagemaker",
        scalable_dimension: str = "sagemaker:variant:DesiredInstanceCount",
    ):
        """Apply the step scaling policy.

        Args:
            client: boto3 Application Auto Scaling client
            resource_id: Resource ID
            service_namespace: Service namespace (default: sagemaker)
            scalable_dimension: Scalable dimension (default: sagemaker:variant:DesiredInstanceCount)

        Raises:
            ClientError: If policy creation fails
        """
        try:
            response = client.put_scaling_policy(
                PolicyName=self.policy_name,
                PolicyType="StepScaling",
                ServiceNamespace=service_namespace,
                ResourceId=resource_id,
                ScalableDimension=scalable_dimension,
                StepScalingPolicyConfiguration={
                    "AdjustmentType": self.adjustment_type,
                    "StepAdjustments": self.step_adjustments,
                    "Cooldown": self.cooldown,
                    "MetricAggregationType": self.metric_aggregation_type,
                },
            )

            logger.info(f"Applied step scaling policy '{self.policy_name}'")
            logger.debug(f"Policy ARN: {response.get('PolicyARN')}")

        except ClientError as e:
            logger.error(f"Failed to apply scaling policy: {e}")
            raise


class AutoScalingManager:
    """Manager for SageMaker endpoint auto-scaling.

    Provides high-level interface for registering scalable targets and
    applying scaling policies to SageMaker endpoints.

    Attributes:
        client: boto3 Application Auto Scaling client
        region: AWS region

    Example:
        >>> manager = AutoScalingManager()
        >>> manager.register_endpoint_variant(
        ...     endpoint_name="my-endpoint",
        ...     variant_name="AllTraffic",
        ...     min_capacity=1,
        ...     max_capacity=10
        ... )
        >>> policy = TargetTrackingPolicy("my-policy", target_value=70.0)
        >>> manager.apply_policy("my-endpoint", "AllTraffic", policy)
    """

    def __init__(
        self,
        region: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
    ):
        """Initialize auto-scaling manager.

        Args:
            region: AWS region (default: from settings)
            aws_access_key: AWS access key (default: from settings)
            aws_secret_key: AWS secret key (default: from settings)

        Raises:
            RuntimeError: If boto3 not available
            ValueError: If AWS credentials not configured
        """
        if not BOTO3_AVAILABLE:
            raise RuntimeError(
                "boto3 is not installed. Cannot create Application Auto Scaling client."
            )

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
            "application-autoscaling",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
        )

    def register_endpoint_variant(
        self,
        endpoint_name: str,
        variant_name: str = "AllTraffic",
        min_capacity: int = 1,
        max_capacity: int = 10,
    ) -> None:
        """Register a SageMaker endpoint variant as a scalable target.

        This is required before applying any scaling policies.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            variant_name: Name of the production variant (default: AllTraffic)
            min_capacity: Minimum instance count (default: 1)
            max_capacity: Maximum instance count (default: 10)

        Example:
            >>> manager.register_endpoint_variant(
            ...     endpoint_name="my-endpoint",
            ...     min_capacity=2,
            ...     max_capacity=20
            ... )
        """
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

        try:
            self.client.register_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                MinCapacity=min_capacity,
                MaxCapacity=max_capacity,
            )

            logger.info(
                f"Registered scalable target for endpoint '{endpoint_name}' "
                f"variant '{variant_name}' (min: {min_capacity}, max: {max_capacity})"
            )

        except ClientError as e:
            logger.error(f"Failed to register scalable target: {e}")
            raise

    def register_inference_component(
        self,
        component_name: str,
        min_capacity: int = 1,
        max_capacity: int = 10,
    ) -> None:
        """Register a SageMaker inference component as a scalable target.

        For endpoints using inference components (multi-model endpoints).

        Args:
            component_name: Name of the inference component
            min_capacity: Minimum copy count (default: 1)
            max_capacity: Maximum copy count (default: 10)

        Example:
            >>> manager.register_inference_component(
            ...     component_name="my-component",
            ...     min_capacity=1,
            ...     max_capacity=6
            ... )
        """
        resource_id = f"inference-component/{component_name}"

        try:
            self.client.register_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:inference-component:DesiredCopyCount",
                MinCapacity=min_capacity,
                MaxCapacity=max_capacity,
            )

            logger.info(
                f"Registered inference component '{component_name}' "
                f"as scalable target (min: {min_capacity}, max: {max_capacity})"
            )

        except ClientError as e:
            logger.error(f"Failed to register inference component: {e}")
            raise

    def apply_policy(
        self,
        endpoint_name: str,
        policy: ScalingPolicy,
        variant_name: str = "AllTraffic",
    ) -> None:
        """Apply a scaling policy to an endpoint variant.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            policy: Scaling policy to apply (TargetTrackingPolicy or StepScalingPolicy)
            variant_name: Name of the production variant (default: AllTraffic)

        Example:
            >>> policy = TargetTrackingPolicy("my-policy", target_value=100.0)
            >>> manager.apply_policy("my-endpoint", policy)
        """
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

        policy.apply(
            client=self.client,
            resource_id=resource_id,
            service_namespace="sagemaker",
            scalable_dimension="sagemaker:variant:DesiredInstanceCount",
        )

    def apply_component_policy(
        self,
        component_name: str,
        policy: ScalingPolicy,
    ) -> None:
        """Apply a scaling policy to an inference component.

        Args:
            component_name: Name of the inference component
            policy: Scaling policy to apply

        Example:
            >>> policy = TargetTrackingPolicy("component-policy", target_value=4.0)
            >>> manager.apply_component_policy("my-component", policy)
        """
        resource_id = f"inference-component/{component_name}"

        # Use different predefined metric type for inference components
        if isinstance(policy, TargetTrackingPolicy):
            policy.predefined_metric_type = "SageMakerInferenceComponentInvocationsPerCopy"

        policy.apply(
            client=self.client,
            resource_id=resource_id,
            service_namespace="sagemaker",
            scalable_dimension="sagemaker:inference-component:DesiredCopyCount",
        )

    def delete_policy(
        self,
        endpoint_name: str,
        policy_name: str,
        variant_name: str = "AllTraffic",
    ) -> None:
        """Delete a scaling policy.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            policy_name: Name of the policy to delete
            variant_name: Name of the production variant (default: AllTraffic)

        Example:
            >>> manager.delete_policy("my-endpoint", "my-policy")
        """
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

        try:
            self.client.delete_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            )

            logger.info(f"Deleted scaling policy '{policy_name}'")

        except ClientError as e:
            logger.error(f"Failed to delete scaling policy: {e}")
            raise

    def deregister_endpoint_variant(
        self,
        endpoint_name: str,
        variant_name: str = "AllTraffic",
    ) -> None:
        """Deregister an endpoint variant as a scalable target.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            variant_name: Name of the production variant (default: AllTraffic)

        Example:
            >>> manager.deregister_endpoint_variant("my-endpoint")
        """
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

        try:
            self.client.deregister_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            )

            logger.info(f"Deregistered scalable target for endpoint '{endpoint_name}'")

        except ClientError as e:
            logger.error(f"Failed to deregister scalable target: {e}")
            raise

    def list_scaling_policies(
        self, endpoint_name: str, variant_name: str = "AllTraffic"
    ) -> list[dict]:
        """List all scaling policies for an endpoint variant.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            variant_name: Name of the production variant (default: AllTraffic)

        Returns:
            list: List of scaling policy dictionaries

        Example:
            >>> policies = manager.list_scaling_policies("my-endpoint")
            >>> for policy in policies:
            ...     print(policy["PolicyName"])
        """
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

        try:
            response = self.client.describe_scaling_policies(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            )

            policies = response.get("ScalingPolicies", [])
            logger.info(f"Found {len(policies)} scaling policies for '{endpoint_name}'")

            return policies

        except ClientError as e:
            logger.error(f"Failed to list scaling policies: {e}")
            return []

    def setup_basic_autoscaling(
        self,
        endpoint_name: str,
        variant_name: str = "AllTraffic",
        min_capacity: int = 1,
        max_capacity: int = 10,
        target_invocations: float = 70.0,
        policy_name: Optional[str] = None,
    ) -> None:
        """Set up basic auto-scaling for an endpoint with sensible defaults.

        Convenience method that registers the target and applies a target tracking policy.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            variant_name: Name of the production variant (default: AllTraffic)
            min_capacity: Minimum instance count (default: 1)
            max_capacity: Maximum instance count (default: 10)
            target_invocations: Target invocations per instance (default: 70.0)
            policy_name: Name for the policy (default: {endpoint_name}-autoscaling)

        Example:
            >>> manager.setup_basic_autoscaling(
            ...     endpoint_name="my-endpoint",
            ...     min_capacity=2,
            ...     max_capacity=20,
            ...     target_invocations=100.0
            ... )
        """
        # Register scalable target
        self.register_endpoint_variant(
            endpoint_name=endpoint_name,
            variant_name=variant_name,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
        )

        # Create and apply target tracking policy
        policy_name = policy_name or f"{endpoint_name}-autoscaling"
        policy = TargetTrackingPolicy(
            policy_name=policy_name,
            target_value=target_invocations,
            predefined_metric_type="SageMakerVariantInvocationsPerInstance",
        )

        self.apply_policy(endpoint_name=endpoint_name, policy=policy, variant_name=variant_name)

        logger.info(
            f"Auto-scaling configured for '{endpoint_name}': "
            f"min={min_capacity}, max={max_capacity}, target={target_invocations}"
        )
