"""Infrastructure as Code utilities for AWS and SageMaker."""

from hermes.infrastructure.aws_roles import (
    create_sagemaker_execution_role,
    create_sagemaker_user,
    get_or_create_execution_role,
)
from hermes.infrastructure.endpoint_manager import (
    SageMakerEndpointManager,
    delete_endpoint_and_config,
)
from hermes.infrastructure.autoscaling import (
    AutoScalingManager,
    ScalingPolicy,
    TargetTrackingPolicy,
)

__all__ = [
    # AWS Roles
    "create_sagemaker_execution_role",
    "create_sagemaker_user",
    "get_or_create_execution_role",
    # Endpoint Management
    "SageMakerEndpointManager",
    "delete_endpoint_and_config",
    # Autoscaling
    "AutoScalingManager",
    "ScalingPolicy",
    "TargetTrackingPolicy",
]
