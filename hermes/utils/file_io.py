"""File I/O utilities for reading and writing various file formats.

This module provides managers for handling different file formats:
- JsonFileManager: JSON file operations
- CsvFileManager: CSV file operations
- ParquetFileManager: Parquet file operations (requires pyarrow)
"""

import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger


class JsonFileManager:
    """Manager for JSON file operations."""

    @classmethod
    def read(cls, filename: str | Path) -> dict | list:
        """Read JSON data from a file.

        Args:
            filename: Path to the JSON file

        Returns:
            Loaded JSON data (dict or list)

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        file_path = Path(filename)

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            logger.debug(f"Successfully read JSON from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File '{file_path}' does not exist.") from None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file: {file_path}")
            raise json.JSONDecodeError(
                msg=f"File '{file_path}' is not properly formatted as JSON.",
                doc=e.doc,
                pos=e.pos,
            ) from None

    @classmethod
    def write(cls, filename: str | Path, data: dict | list, indent: int = 4) -> Path:
        """Write data to a JSON file.

        Args:
            filename: Path to the output JSON file
            data: Data to write (must be JSON serializable)
            indent: Number of spaces for indentation (default: 4)

        Returns:
            Absolute path to the written file
        """
        file_path = Path(filename).resolve().absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with file_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=indent, ensure_ascii=False)
            logger.info(f"Successfully wrote JSON to {file_path}")
            return file_path
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize data to JSON: {e}")
            raise

    @classmethod
    def append(cls, filename: str | Path, data: dict | list) -> Path:
        """Append data to an existing JSON array file.

        If the file doesn't exist or is empty, creates a new array.
        The file must contain a JSON array (list).

        Args:
            filename: Path to the JSON file
            data: Data to append (single item or list)

        Returns:
            Absolute path to the written file
        """
        file_path = Path(filename)

        # Read existing data
        if file_path.exists() and file_path.stat().st_size > 0:
            try:
                existing_data = cls.read(file_path)
                if not isinstance(existing_data, list):
                    raise ValueError("Can only append to JSON arrays (lists)")
            except json.JSONDecodeError:
                logger.warning(f"Existing file {file_path} is not valid JSON, overwriting")
                existing_data = []
        else:
            existing_data = []

        # Append new data
        if isinstance(data, list):
            existing_data.extend(data)
        else:
            existing_data.append(data)

        # Write back
        return cls.write(file_path, existing_data)


class CsvFileManager:
    """Manager for CSV file operations."""

    @classmethod
    def read(
        cls,
        filename: str | Path,
        delimiter: str = ",",
        skip_header: bool = False,
    ) -> list[dict[str, Any]]:
        """Read CSV data from a file.

        Args:
            filename: Path to the CSV file
            delimiter: CSV delimiter (default: ',')
            skip_header: Whether to skip the first row (default: False)

        Returns:
            List of dictionaries representing rows

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(filename)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        try:
            with file_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file, delimiter=delimiter)
                if skip_header:
                    next(reader, None)
                data = list(reader)
            logger.debug(f"Successfully read {len(data)} rows from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to read CSV file {file_path}: {e}")
            raise

    @classmethod
    def write(
        cls,
        filename: str | Path,
        data: list[dict[str, Any]],
        delimiter: str = ",",
        fieldnames: list[str] | None = None,
    ) -> Path:
        """Write data to a CSV file.

        Args:
            filename: Path to the output CSV file
            data: List of dictionaries to write
            delimiter: CSV delimiter (default: ',')
            fieldnames: Column names (default: keys from first dict)

        Returns:
            Absolute path to the written file
        """
        file_path = Path(filename).resolve().absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not data:
            logger.warning(f"No data to write to {file_path}")
            file_path.touch()
            return file_path

        # Use provided fieldnames or extract from first dict
        if fieldnames is None:
            fieldnames = list(data[0].keys())

        try:
            with file_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"Successfully wrote {len(data)} rows to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to write CSV file {file_path}: {e}")
            raise


class ParquetFileManager:
    """Manager for Parquet file operations.

    Requires pyarrow to be installed.
    """

    @classmethod
    def read(cls, filename: str | Path) -> list[dict[str, Any]]:
        """Read Parquet data from a file.

        Args:
            filename: Path to the Parquet file

        Returns:
            List of dictionaries representing rows

        Raises:
            ImportError: If pyarrow is not installed
            FileNotFoundError: If the file doesn't exist
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet operations. Install with: pip install pyarrow")

        file_path = Path(filename)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        try:
            table = pq.read_table(file_path)
            data = table.to_pylist()
            logger.debug(f"Successfully read {len(data)} rows from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to read Parquet file {file_path}: {e}")
            raise

    @classmethod
    def write(cls, filename: str | Path, data: list[dict[str, Any]]) -> Path:
        """Write data to a Parquet file.

        Args:
            filename: Path to the output Parquet file
            data: List of dictionaries to write

        Returns:
            Absolute path to the written file

        Raises:
            ImportError: If pyarrow is not installed
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet operations. Install with: pip install pyarrow")

        file_path = Path(filename).resolve().absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not data:
            logger.warning(f"No data to write to {file_path}")
            return file_path

        try:
            table = pa.Table.from_pylist(data)
            pq.write_table(table, file_path)
            logger.info(f"Successfully wrote {len(data)} rows to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to write Parquet file {file_path}: {e}")
            raise


def get_file_manager(filename: str | Path):
    """Get the appropriate file manager based on file extension.

    Args:
        filename: Path to the file

    Returns:
        File manager class (JsonFileManager, CsvFileManager, or ParquetFileManager)

    Raises:
        ValueError: If the file extension is not supported
    """
    file_path = Path(filename)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        return JsonFileManager
    elif suffix == ".csv":
        return CsvFileManager
    elif suffix == ".parquet":
        return ParquetFileManager
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Supported: .json, .csv, .parquet")

