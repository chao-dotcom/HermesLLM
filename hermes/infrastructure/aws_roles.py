"""AWS IAM role management utilities for SageMaker.

This module provides utilities for creating and managing AWS IAM roles required
for SageMaker training and deployment operations.
"""

import json
from pathlib import Path
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


def _get_iam_client():
    """Get configured IAM client.

    Returns:
        boto3.client: Configured IAM client

    Raises:
        RuntimeError: If boto3 is not available or AWS credentials not configured
    """
    if not BOTO3_AVAILABLE:
        raise RuntimeError("boto3 is not installed. Cannot create IAM client.")

    if not settings.AWS_REGION:
        raise ValueError("AWS_REGION is not set in settings")
    if not settings.AWS_ACCESS_KEY:
        raise ValueError("AWS_ACCESS_KEY is not set in settings")
    if not settings.AWS_SECRET_KEY:
        raise ValueError("AWS_SECRET_KEY is not set in settings")

    return boto3.client(
        "iam",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )


def create_sagemaker_user(
    username: str, save_credentials: bool = True, output_file: Optional[Path] = None
) -> dict[str, str]:
    """Create an IAM user with SageMaker permissions.

    Creates a new IAM user with full SageMaker access and related AWS service
    permissions. Optionally saves the generated access credentials to a file.

    Args:
        username: Name for the new IAM user
        save_credentials: Whether to save credentials to file (default: True)
        output_file: Path to save credentials JSON (default: sagemaker_user_credentials.json)

    Returns:
        dict: Dictionary with AccessKeyId and SecretAccessKey

    Raises:
        RuntimeError: If boto3 not available
        ValueError: If AWS credentials not configured
        ClientError: If IAM operations fail

    Example:
        >>> credentials = create_sagemaker_user("sagemaker-deployer")
        >>> print(credentials["AccessKeyId"])
    """
    iam = _get_iam_client()

    try:
        # Create user
        iam.create_user(UserName=username)
        logger.info(f"Created IAM user: {username}")

        # Attach necessary policies for SageMaker operations
        policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AWSCloudFormationFullAccess",
            "arn:aws:iam::aws:policy/IAMFullAccess",
            "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        ]

        for policy in policies:
            iam.attach_user_policy(UserName=username, PolicyArn=policy)
            logger.debug(f"Attached policy: {policy}")

        # Create access key
        response = iam.create_access_key(UserName=username)
        access_key = response["AccessKey"]

        credentials = {
            "AccessKeyId": access_key["AccessKeyId"],
            "SecretAccessKey": access_key["SecretAccessKey"],
        }

        logger.info(f"User '{username}' created successfully with access credentials")

        # Save credentials if requested
        if save_credentials:
            output_path = output_file or Path("sagemaker_user_credentials.json")
            with output_path.open("w") as f:
                json.dump(credentials, f, indent=2)
            logger.info(f"Credentials saved to: {output_path}")

        return credentials

    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            logger.warning(f"User '{username}' already exists")
            raise
        else:
            logger.error(f"Failed to create user: {e}")
            raise


def create_sagemaker_execution_role(
    role_name: str,
    save_arn: bool = True,
    output_file: Optional[Path] = None,
    additional_policies: Optional[list[str]] = None,
) -> str:
    """Create an IAM execution role for SageMaker.

    Creates an IAM role with trust relationship for SageMaker service and
    attaches necessary policies for training and deployment operations.

    Args:
        role_name: Name for the execution role
        save_arn: Whether to save role ARN to file (default: True)
        output_file: Path to save ARN JSON (default: sagemaker_execution_role.json)
        additional_policies: Optional list of additional policy ARNs to attach

    Returns:
        str: ARN of the created or existing role

    Raises:
        RuntimeError: If boto3 not available
        ValueError: If AWS credentials not configured
        ClientError: If IAM operations fail

    Example:
        >>> role_arn = create_sagemaker_execution_role("SageMakerExecutionRole")
        >>> print(role_arn)
        arn:aws:iam::123456789012:role/SageMakerExecutionRole
    """
    iam = _get_iam_client()

    # Define trust relationship policy for SageMaker
    trust_relationship = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        # Create the IAM role
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_relationship),
            Description="Execution role for SageMaker training and deployment",
        )
        role_arn = role["Role"]["Arn"]
        logger.info(f"Created IAM role: {role_name}")
        logger.info(f"Role ARN: {role_arn}")

    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            logger.warning(f"Role '{role_name}' already exists. Fetching ARN...")
            role = iam.get_role(RoleName=role_name)
            role_arn = role["Role"]["Arn"]
            logger.info(f"Existing role ARN: {role_arn}")
        else:
            logger.error(f"Failed to create role: {e}")
            raise

    # Attach base policies required for SageMaker
    base_policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
    ]

    all_policies = base_policies + (additional_policies or [])

    for policy in all_policies:
        try:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)
            logger.debug(f"Attached policy: {policy}")
        except ClientError as e:
            # Policy might already be attached
            if e.response["Error"]["Code"] != "EntityAlreadyExists":
                logger.warning(f"Failed to attach policy {policy}: {e}")

    # Save role ARN if requested
    if save_arn:
        output_path = output_file or Path("sagemaker_execution_role.json")
        with output_path.open("w") as f:
            json.dump({"RoleArn": role_arn}, f, indent=2)
        logger.info(f"Role ARN saved to: {output_path}")

    return role_arn


def get_or_create_execution_role(
    role_name: Optional[str] = None,
    additional_policies: Optional[list[str]] = None,
) -> str:
    """Get existing execution role ARN or create new one.

    Convenience function that checks settings for an existing role ARN,
    or creates a new role if not configured.

    Args:
        role_name: Name for new role if creation needed (default: "SageMakerExecutionRoleHermes")
        additional_policies: Optional additional policy ARNs to attach

    Returns:
        str: ARN of the execution role

    Example:
        >>> role_arn = get_or_create_execution_role()
        >>> # Use role_arn in SageMaker operations
    """
    # Check if role ARN already configured in settings
    if settings.AWS_ARN_ROLE:
        logger.info(f"Using configured execution role: {settings.AWS_ARN_ROLE}")
        return settings.AWS_ARN_ROLE

    # Create new role
    role_name = role_name or "SageMakerExecutionRoleHermes"
    logger.info(f"No execution role configured. Creating: {role_name}")

    return create_sagemaker_execution_role(
        role_name=role_name,
        save_arn=True,
        additional_policies=additional_policies,
    )


def delete_role(role_name: str, detach_policies: bool = True) -> None:
    """Delete an IAM role.

    Optionally detaches all managed policies before deletion.

    Args:
        role_name: Name of the role to delete
        detach_policies: Whether to detach policies first (default: True)

    Raises:
        RuntimeError: If boto3 not available
        ValueError: If AWS credentials not configured
        ClientError: If IAM operations fail

    Example:
        >>> delete_role("OldSageMakerRole")
    """
    iam = _get_iam_client()

    try:
        if detach_policies:
            # List and detach all attached managed policies
            response = iam.list_attached_role_policies(RoleName=role_name)
            for policy in response.get("AttachedPolicies", []):
                iam.detach_role_policy(RoleName=role_name, PolicyArn=policy["PolicyArn"])
                logger.debug(f"Detached policy: {policy['PolicyArn']}")

        # Delete the role
        iam.delete_role(RoleName=role_name)
        logger.info(f"Deleted IAM role: {role_name}")

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            logger.warning(f"Role '{role_name}' does not exist")
        else:
            logger.error(f"Failed to delete role: {e}")
            raise


def list_sagemaker_roles() -> list[dict]:
    """List all IAM roles with SageMaker in the name.

    Returns:
        list: List of role dictionaries with RoleName and Arn

    Example:
        >>> roles = list_sagemaker_roles()
        >>> for role in roles:
        ...     print(f"{role['RoleName']}: {role['Arn']}")
    """
    if not BOTO3_AVAILABLE:
        logger.warning("boto3 not available. Cannot list roles.")
        return []

    iam = _get_iam_client()

    try:
        response = iam.list_roles()
        sagemaker_roles = [
            {"RoleName": role["RoleName"], "Arn": role["Arn"]}
            for role in response.get("Roles", [])
            if "sagemaker" in role["RoleName"].lower()
        ]

        logger.info(f"Found {len(sagemaker_roles)} SageMaker-related roles")
        return sagemaker_roles

    except ClientError as e:
        logger.error(f"Failed to list roles: {e}")
        return []
