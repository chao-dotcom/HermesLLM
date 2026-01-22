"""ZenML pipeline for exporting artifacts to JSON files.

This module provides a pipeline for exporting ZenML artifacts (raw_documents,
cleaned_documents, chunks, datasets, etc.) to JSON files for external use.
"""

from pathlib import Path

from loguru import logger

try:
    from zenml import pipeline
    from zenml.client import Client

    ZENML_AVAILABLE = True
except ImportError:
    logger.warning("ZenML not available. Artifact export pipeline will not work.")
    ZENML_AVAILABLE = False
    # Dummy decorator
    def pipeline(func):
        return func

    Client = None

from hermes.zenml_steps.export_steps import export_artifact_to_json


if ZENML_AVAILABLE:

    @pipeline
    def artifact_export_pipeline(
        artifact_names: list[str],
        output_dir: Path = Path("output/artifacts"),
    ) -> None:
        """Export ZenML artifacts to JSON files.

        This pipeline retrieves artifacts from the ZenML artifact store
        and exports them to JSON files for external use, analysis, or backup.

        Args:
            artifact_names: List of artifact names/IDs to export
            output_dir: Directory for output JSON files (default: output/artifacts)
        """
        logger.info(f"Starting artifact export pipeline for {len(artifact_names)} artifacts")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        client = Client()

        for artifact_name in artifact_names:
            try:
                # Retrieve artifact from ZenML
                logger.info(f"Retrieving artifact: {artifact_name}")
                artifact = client.get_artifact_version(name_id_or_prefix=artifact_name)

                # Export to JSON
                output_file = output_path / f"{artifact_name}.json"
                export_artifact_to_json(
                    artifact=artifact.load(),
                    artifact_name=artifact_name,
                    output_file=output_file,
                )

                logger.info(f"Exported artifact '{artifact_name}' to {output_file}")

            except Exception as e:
                logger.error(f"Failed to export artifact '{artifact_name}': {e}")
                continue

        logger.info(f"Artifact export pipeline completed. Output directory: {output_path}")

    @pipeline
    def multi_artifact_export_pipeline(
        artifact_configs: list[dict],
    ) -> None:
        """Export multiple artifacts with custom configurations.

        Args:
            artifact_configs: List of dicts with 'name', 'output_file', and optional 'format'
                             Example: [
                                 {'name': 'raw_documents', 'output_file': 'data/raw.json'},
                                 {'name': 'cleaned_documents', 'output_file': 'data/cleaned.json'},
                             ]
        """
        logger.info(f"Starting multi-artifact export pipeline for {len(artifact_configs)} artifacts")

        client = Client()

        for config in artifact_configs:
            artifact_name = config.get("name")
            output_file = config.get("output_file")

            if not artifact_name or not output_file:
                logger.warning(f"Skipping invalid config: {config}")
                continue

            try:
                # Retrieve artifact
                logger.info(f"Retrieving artifact: {artifact_name}")
                artifact = client.get_artifact_version(name_id_or_prefix=artifact_name)

                # Export to JSON
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                export_artifact_to_json(
                    artifact=artifact.load(),
                    artifact_name=artifact_name,
                    output_file=output_path,
                )

                logger.info(f"Exported artifact '{artifact_name}' to {output_path}")

            except Exception as e:
                logger.error(f"Failed to export artifact '{artifact_name}': {e}")
                continue

        logger.info("Multi-artifact export pipeline completed")

else:

    def artifact_export_pipeline(artifact_names: list[str], output_dir: Path = Path("output/artifacts")) -> None:
        """Fallback when ZenML is not available."""
        logger.error("ZenML not available. Cannot run artifact export pipeline.")
        raise ImportError("ZenML is required for artifact export pipeline")

    def multi_artifact_export_pipeline(artifact_configs: list[dict]) -> None:
        """Fallback when ZenML is not available."""
        logger.error("ZenML not available. Cannot run multi-artifact export pipeline.")
        raise ImportError("ZenML is required for multi-artifact export pipeline")


def export_artifacts_from_run(
    run_name: str,
    output_dir: Path = Path("output/artifacts"),
    artifact_filter: list[str] | None = None,
) -> dict[str, Path]:
    """Export all artifacts from a specific ZenML pipeline run.

    This is a helper function (not a ZenML pipeline) that exports artifacts
    from a completed pipeline run.

    Args:
        run_name: Name or ID of the pipeline run
        output_dir: Directory for output JSON files
        artifact_filter: Optional list of artifact names to export (exports all if None)

    Returns:
        Dictionary mapping artifact names to exported file paths
    """
    if not ZENML_AVAILABLE:
        logger.error("ZenML not available. Cannot export artifacts from run.")
        raise ImportError("ZenML is required for this function")

    logger.info(f"Exporting artifacts from run: {run_name}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = Client()
    exported_files = {}

    try:
        # Get pipeline run
        run = client.get_pipeline_run(run_name)

        # Get all artifacts from the run
        for step_name, step in run.steps.items():
            for output_name, artifact in step.outputs.items():
                artifact_name = artifact.name

                # Apply filter if provided
                if artifact_filter and artifact_name not in artifact_filter:
                    logger.debug(f"Skipping artifact '{artifact_name}' (not in filter)")
                    continue

                try:
                    # Export artifact
                    output_file = output_path / f"{artifact_name}.json"
                    from hermes.zenml_steps.export_steps import export_artifact_to_json

                    export_artifact_to_json(
                        artifact=artifact.load(),
                        artifact_name=artifact_name,
                        output_file=output_file,
                    )

                    exported_files[artifact_name] = output_file
                    logger.info(f"Exported artifact '{artifact_name}' to {output_file}")

                except Exception as e:
                    logger.error(f"Failed to export artifact '{artifact_name}': {e}")
                    continue

        logger.info(f"Exported {len(exported_files)} artifacts from run '{run_name}'")
        return exported_files

    except Exception as e:
        logger.error(f"Failed to export artifacts from run '{run_name}': {e}")
        raise

