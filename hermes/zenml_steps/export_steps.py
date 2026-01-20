"""
Export Steps for ZenML Pipelines

This module contains steps for exporting artifacts and results.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from typing_extensions import Annotated

from loguru import logger
from zenml import get_step_context, step


@step
def export_to_json(
    data: Dict[str, Any],
    output_path: str,
    pretty: bool = True,
) -> Annotated[str, "output_file_path"]:
    """
    Export data to JSON file.
    
    Args:
        data: Data to export
        output_path: Output file path
        pretty: Whether to format JSON with indentation
        
    Returns:
        Path to exported file
    """
    logger.info(f"Exporting data to {output_path}")
    
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)
        
        logger.success(f"Exported to {output_path}")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="output_file_path",
            metadata={
                "output_path": output_path,
                "file_size_bytes": output_file.stat().st_size,
            }
        )
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return f"Failed: {str(e)}"


@step
def collect_pipeline_artifacts(
    invocation_ids: List[str],
) -> Annotated[Dict[str, Any], "collected_artifacts"]:
    """
    Collect artifacts from multiple pipeline steps.
    
    Args:
        invocation_ids: List of step invocation IDs
        
    Returns:
        Dictionary of collected artifacts
    """
    logger.info(f"Collecting artifacts from {len(invocation_ids)} steps")
    
    try:
        # In a real implementation, this would query ZenML artifact store
        # For now, return a summary
        artifacts = {
            "invocation_ids": invocation_ids,
            "num_artifacts": len(invocation_ids),
        }
        
        logger.success(f"Collected {len(invocation_ids)} artifacts")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="collected_artifacts",
            metadata=artifacts
        )
        
        return artifacts
        
    except Exception as e:
        logger.error(f"Artifact collection failed: {e}")
        return {"error": str(e)}
