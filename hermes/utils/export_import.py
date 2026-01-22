"""Export/Import utilities for data serialization and deserialization.

This module provides utilities for exporting and importing data artifacts,
including serialization of complex objects, ZenML artifacts, and database documents.
"""

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from hermes.utils.file_io import JsonFileManager, get_file_manager


class ArtifactSerializer:
    """Serializer for converting complex objects to JSON-compatible formats."""

    @staticmethod
    def serialize(artifact: Any) -> dict | list | str | int | float | bool | None:
        """Serialize an artifact to a JSON-compatible format.

        Handles:
        - Pydantic models (BaseModel)
        - Lists and dicts (recursive)
        - Primitive types (str, int, float, bool, None)

        Args:
            artifact: Object to serialize

        Returns:
            JSON-compatible representation
        """
        if artifact is None:
            return None
        elif isinstance(artifact, list):
            return [ArtifactSerializer.serialize(item) for item in artifact]
        elif isinstance(artifact, dict):
            return {key: ArtifactSerializer.serialize(value) for key, value in artifact.items()}
        elif isinstance(artifact, BaseModel):
            return artifact.model_dump()
        elif isinstance(artifact, (str, int, float, bool)):
            return artifact
        else:
            # Try to convert to string as fallback
            logger.warning(f"Unknown type {type(artifact)}, converting to string")
            return str(artifact)

    @staticmethod
    def deserialize(data: Any, target_class: type[BaseModel] | None = None) -> Any:
        """Deserialize JSON data back to objects.

        Args:
            data: JSON-compatible data
            target_class: Optional Pydantic model class to deserialize into

        Returns:
            Deserialized object
        """
        if data is None:
            return None
        elif isinstance(data, list):
            return [ArtifactSerializer.deserialize(item, target_class) for item in data]
        elif isinstance(data, dict):
            if target_class and issubclass(target_class, BaseModel):
                return target_class(**data)
            else:
                return {key: ArtifactSerializer.deserialize(value) for key, value in data.items()}
        else:
            return data


class DataExporter:
    """Export data to various file formats."""

    @staticmethod
    def export_to_json(
        data: Any,
        output_file: str | Path,
        serialize: bool = True,
    ) -> Path:
        """Export data to a JSON file.

        Args:
            data: Data to export
            output_file: Path to output JSON file
            serialize: Whether to serialize complex objects (default: True)

        Returns:
            Absolute path to the exported file
        """
        if serialize:
            serialized_data = ArtifactSerializer.serialize(data)
        else:
            serialized_data = data

        # Wrap non-dict data
        if not isinstance(serialized_data, dict):
            serialized_data = {"data": serialized_data}

        output_path = JsonFileManager.write(output_file, serialized_data)
        logger.info(f"Exported data to {output_path}")
        return output_path

    @staticmethod
    def export_to_csv(
        data: list[dict],
        output_file: str | Path,
        fieldnames: list[str] | None = None,
    ) -> Path:
        """Export data to a CSV file.

        Args:
            data: List of dictionaries to export
            output_file: Path to output CSV file
            fieldnames: Optional list of field names for CSV header

        Returns:
            Absolute path to the exported file
        """
        from hermes.utils.file_io import CsvFileManager

        output_path = CsvFileManager.write(output_file, data, fieldnames=fieldnames)
        logger.info(f"Exported {len(data)} rows to {output_path}")
        return output_path

    @staticmethod
    def export_to_parquet(data: list[dict], output_file: str | Path) -> Path:
        """Export data to a Parquet file.

        Args:
            data: List of dictionaries to export
            output_file: Path to output Parquet file

        Returns:
            Absolute path to the exported file
        """
        from hermes.utils.file_io import ParquetFileManager

        output_path = ParquetFileManager.write(output_file, data)
        logger.info(f"Exported {len(data)} rows to {output_path}")
        return output_path

    @staticmethod
    def export_documents(
        documents: list[BaseModel],
        output_file: str | Path,
        file_format: str = "json",
    ) -> Path:
        """Export a list of Pydantic documents to a file.

        Args:
            documents: List of Pydantic model instances
            output_file: Path to output file
            file_format: File format ('json', 'csv', 'parquet')

        Returns:
            Absolute path to the exported file
        """
        # Serialize documents
        serialized_docs = [doc.model_dump() for doc in documents]

        # Export based on format
        if file_format == "json":
            return DataExporter.export_to_json(serialized_docs, output_file, serialize=False)
        elif file_format == "csv":
            return DataExporter.export_to_csv(serialized_docs, output_file)
        elif file_format == "parquet":
            return DataExporter.export_to_parquet(serialized_docs, output_file)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


class DataImporter:
    """Import data from various file formats."""

    @staticmethod
    def import_from_json(
        input_file: str | Path,
        target_class: type[BaseModel] | None = None,
    ) -> Any:
        """Import data from a JSON file.

        Args:
            input_file: Path to input JSON file
            target_class: Optional Pydantic model class to deserialize into

        Returns:
            Imported data
        """
        data = JsonFileManager.read(input_file)
        logger.info(f"Imported data from {input_file}")

        if target_class:
            return ArtifactSerializer.deserialize(data, target_class)
        return data

    @staticmethod
    def import_from_csv(input_file: str | Path) -> list[dict]:
        """Import data from a CSV file.

        Args:
            input_file: Path to input CSV file

        Returns:
            List of dictionaries representing rows
        """
        from hermes.utils.file_io import CsvFileManager

        data = CsvFileManager.read(input_file)
        logger.info(f"Imported {len(data)} rows from {input_file}")
        return data

    @staticmethod
    def import_from_parquet(input_file: str | Path) -> list[dict]:
        """Import data from a Parquet file.

        Args:
            input_file: Path to input Parquet file

        Returns:
            List of dictionaries representing rows
        """
        from hermes.utils.file_io import ParquetFileManager

        data = ParquetFileManager.read(input_file)
        logger.info(f"Imported {len(data)} rows from {input_file}")
        return data

    @staticmethod
    def import_documents(
        input_file: str | Path,
        target_class: type[BaseModel],
        file_format: str = "json",
    ) -> list[BaseModel]:
        """Import a list of Pydantic documents from a file.

        Args:
            input_file: Path to input file
            target_class: Pydantic model class to deserialize into
            file_format: File format ('json', 'csv', 'parquet')

        Returns:
            List of Pydantic model instances
        """
        # Import based on format
        if file_format == "json":
            data = DataImporter.import_from_json(input_file)
        elif file_format == "csv":
            data = DataImporter.import_from_csv(input_file)
        elif file_format == "parquet":
            data = DataImporter.import_from_parquet(input_file)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Handle both direct list and wrapped data
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, list):
            raise ValueError(f"Expected list of documents, got {type(data)}")

        # Deserialize to Pydantic models
        documents = [target_class(**item) if isinstance(item, dict) else item for item in data]
        logger.info(f"Imported {len(documents)} {target_class.__name__} documents")
        return documents


class BatchExporter:
    """Export data in batches for large datasets."""

    @staticmethod
    def export_batches(
        data: list[Any],
        output_dir: str | Path,
        batch_size: int = 1000,
        file_prefix: str = "batch",
        file_format: str = "json",
    ) -> list[Path]:
        """Export data in batches to multiple files.

        Args:
            data: List of data items to export
            output_dir: Directory for output files
            batch_size: Number of items per batch
            file_prefix: Prefix for batch filenames
            file_format: File format ('json', 'csv', 'parquet')

        Returns:
            List of paths to created batch files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        batch_files = []
        total_batches = (len(data) + batch_size - 1) // batch_size

        for i in range(0, len(data), batch_size):
            batch_num = i // batch_size + 1
            batch_data = data[i : i + batch_size]

            # Generate filename
            filename = f"{file_prefix}_{batch_num:04d}.{file_format}"
            batch_file = output_path / filename

            # Export batch
            if file_format == "json":
                DataExporter.export_to_json(batch_data, batch_file)
            elif file_format == "csv":
                DataExporter.export_to_csv(batch_data, batch_file)
            elif file_format == "parquet":
                DataExporter.export_to_parquet(batch_data, batch_file)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            batch_files.append(batch_file)
            logger.info(f"Exported batch {batch_num}/{total_batches} to {batch_file}")

        logger.info(f"Exported {len(data)} items in {len(batch_files)} batches to {output_path}")
        return batch_files

    @staticmethod
    def import_batches(
        input_dir: str | Path,
        file_pattern: str = "batch_*.json",
        target_class: type[BaseModel] | None = None,
    ) -> list[Any]:
        """Import data from multiple batch files.

        Args:
            input_dir: Directory containing batch files
            file_pattern: Glob pattern for batch files
            target_class: Optional Pydantic model class to deserialize into

        Returns:
            Combined list of all imported data
        """
        input_path = Path(input_dir)
        batch_files = sorted(input_path.glob(file_pattern))

        if not batch_files:
            logger.warning(f"No batch files found in {input_path} matching {file_pattern}")
            return []

        all_data = []
        for batch_file in batch_files:
            # Determine format from extension
            file_format = batch_file.suffix[1:]  # Remove leading dot

            if file_format == "json":
                batch_data = DataImporter.import_from_json(batch_file, target_class)
            elif file_format == "csv":
                batch_data = DataImporter.import_from_csv(batch_file)
            elif file_format == "parquet":
                batch_data = DataImporter.import_from_parquet(batch_file)
            else:
                logger.warning(f"Skipping file with unsupported format: {batch_file}")
                continue

            # Handle wrapped data
            if isinstance(batch_data, dict) and "data" in batch_data:
                batch_data = batch_data["data"]

            if isinstance(batch_data, list):
                all_data.extend(batch_data)
            else:
                all_data.append(batch_data)

        logger.info(f"Imported {len(all_data)} items from {len(batch_files)} batch files")
        return all_data

