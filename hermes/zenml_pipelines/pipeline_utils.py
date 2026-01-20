"""
Pipeline Utilities

Helper functions for ZenML pipeline configuration and execution.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from loguru import logger


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def generate_run_name(pipeline_name: str, timestamp: bool = True) -> str:
    """
    Generate a unique run name for a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        timestamp: Whether to include timestamp
        
    Returns:
        Unique run name
    """
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{pipeline_name}_{ts}"
    return pipeline_name


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that config has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key names
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged


def save_pipeline_output(
    output_data: Any,
    output_dir: str,
    filename: str,
    format: str = "json",
) -> str:
    """
    Save pipeline output to file.
    
    Args:
        output_data: Data to save
        output_dir: Output directory
        filename: Output filename
        format: Output format (json, yaml, txt)
        
    Returns:
        Path to saved file
    """
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / filename
    
    if format == "json":
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
    elif format == "yaml":
        with open(output_file, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False)
    elif format == "txt":
        with open(output_file, 'w') as f:
            f.write(str(output_data))
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved output to {output_file}")
    return str(output_file)


def get_cache_key(pipeline_name: str, params: Dict[str, Any]) -> str:
    """
    Generate cache key for pipeline run.
    
    Args:
        pipeline_name: Name of pipeline
        params: Pipeline parameters
        
    Returns:
        Cache key string
    """
    import hashlib
    import json
    
    # Sort params for consistent hashing
    sorted_params = json.dumps(params, sort_keys=True, default=str)
    hash_obj = hashlib.sha256(f"{pipeline_name}:{sorted_params}".encode())
    return hash_obj.hexdigest()[:16]
