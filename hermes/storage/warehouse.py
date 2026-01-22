"""Data warehouse export and import utilities.

This module provides tools for exporting and importing data from MongoDB
collections to various file formats (JSON, CSV, Parquet).
"""

from pathlib import Path
from typing import Type

from loguru import logger
from pydantic import BaseModel

from hermes.core.documents import RawDocument
from hermes.core.cleaned_documents import CleanedDocument
from hermes.core.chunks import Chunk, EmbeddedChunk
from hermes.storage.database import MongoDBManager
from hermes.utils.export_import import DataExporter, DataImporter


class DataWarehouseExporter:
    """Export data from MongoDB collections to files."""

    def __init__(self, mongodb_manager: MongoDBManager | None = None):
        """Initialize the data warehouse exporter.

        Args:
            mongodb_manager: MongoDB manager instance (creates new if None)
        """
        self.mongodb = mongodb_manager or MongoDBManager()

    def export_collection(
        self,
        collection_name: str,
        output_file: str | Path,
        file_format: str = "json",
        query: dict | None = None,
        limit: int | None = None,
    ) -> Path:
        """Export a MongoDB collection to a file.

        Args:
            collection_name: Name of the MongoDB collection
            output_file: Path to output file
            file_format: File format ('json', 'csv', 'parquet')
            query: Optional MongoDB query filter
            limit: Optional limit on number of documents

        Returns:
            Path to the exported file
        """
        logger.info(f"Exporting collection '{collection_name}' to {output_file}")

        # Fetch documents from MongoDB
        documents = self.mongodb.find_documents(
            collection_name=collection_name,
            query=query or {},
            limit=limit,
        )

        if not documents:
            logger.warning(f"No documents found in collection '{collection_name}'")
            # Create empty file
            Path(output_file).touch()
            return Path(output_file)

        # Export to file
        if file_format == "json":
            output_path = DataExporter.export_to_json(documents, output_file, serialize=False)
        elif file_format == "csv":
            output_path = DataExporter.export_to_csv(documents, output_file)
        elif file_format == "parquet":
            output_path = DataExporter.export_to_parquet(documents, output_file)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Exported {len(documents)} documents from '{collection_name}' to {output_path}")
        return output_path

    def export_all_collections(
        self,
        output_dir: str | Path,
        file_format: str = "json",
        collections: list[str] | None = None,
    ) -> dict[str, Path]:
        """Export all or specified MongoDB collections to files.

        Args:
            output_dir: Directory for output files
            file_format: File format ('json', 'csv', 'parquet')
            collections: Optional list of collection names (exports all if None)

        Returns:
            Dictionary mapping collection names to exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get collections to export
        if collections is None:
            collections = self.mongodb.list_collections()

        exported_files = {}

        for collection_name in collections:
            output_file = output_path / f"{collection_name}.{file_format}"
            try:
                exported_path = self.export_collection(
                    collection_name=collection_name,
                    output_file=output_file,
                    file_format=file_format,
                )
                exported_files[collection_name] = exported_path
            except Exception as e:
                logger.error(f"Failed to export collection '{collection_name}': {e}")
                continue

        logger.info(f"Exported {len(exported_files)} collections to {output_path}")
        return exported_files

    def export_documents_by_type(
        self,
        output_dir: str | Path,
        file_format: str = "json",
    ) -> dict[str, Path]:
        """Export documents by type (raw, cleaned, chunks, embeddings).

        Args:
            output_dir: Directory for output files
            file_format: File format ('json', 'csv', 'parquet')

        Returns:
            Dictionary mapping document types to exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}
        document_types = {
            "raw_documents": "raw_documents",
            "cleaned_documents": "cleaned_documents",
            "chunks": "chunks",
            "embedded_chunks": "embedded_chunks",
        }

        for doc_type, collection_name in document_types.items():
            try:
                output_file = output_path / f"{doc_type}.{file_format}"
                exported_path = self.export_collection(
                    collection_name=collection_name,
                    output_file=output_file,
                    file_format=file_format,
                )
                exported_files[doc_type] = exported_path
            except Exception as e:
                logger.warning(f"Failed to export {doc_type}: {e}")
                continue

        return exported_files


class DataWarehouseImporter:
    """Import data from files into MongoDB collections."""

    def __init__(self, mongodb_manager: MongoDBManager | None = None):
        """Initialize the data warehouse importer.

        Args:
            mongodb_manager: MongoDB manager instance (creates new if None)
        """
        self.mongodb = mongodb_manager or MongoDBManager()

    def import_collection(
        self,
        collection_name: str,
        input_file: str | Path,
        file_format: str = "json",
        document_class: Type[BaseModel] | None = None,
        batch_size: int = 1000,
    ) -> int:
        """Import data from a file into a MongoDB collection.

        Args:
            collection_name: Name of the MongoDB collection
            input_file: Path to input file
            file_format: File format ('json', 'csv', 'parquet')
            document_class: Optional Pydantic model class for validation
            batch_size: Number of documents to insert per batch

        Returns:
            Number of documents imported
        """
        logger.info(f"Importing data from {input_file} to collection '{collection_name}'")

        # Read data from file
        if file_format == "json":
            data = DataImporter.import_from_json(input_file)
        elif file_format == "csv":
            data = DataImporter.import_from_csv(input_file)
        elif file_format == "parquet":
            data = DataImporter.import_from_parquet(input_file)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Handle wrapped data
        if isinstance(data, dict):
            if "data" in data:
                data = data["data"]
            else:
                # Single document
                data = [data]

        if not isinstance(data, list):
            raise ValueError(f"Expected list of documents, got {type(data)}")

        if not data:
            logger.warning(f"No data to import from {input_file}")
            return 0

        # Validate with document class if provided
        if document_class:
            validated_docs = []
            for item in data:
                try:
                    if isinstance(item, dict):
                        doc = document_class(**item)
                        validated_docs.append(doc.model_dump())
                    else:
                        validated_docs.append(item)
                except Exception as e:
                    logger.warning(f"Skipping invalid document: {e}")
                    continue
            data = validated_docs

        # Insert in batches
        total_inserted = 0
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            self.mongodb.insert_many(collection_name, batch)
            total_inserted += len(batch)
            logger.debug(f"Inserted batch {i // batch_size + 1} ({len(batch)} documents)")

        logger.info(f"Imported {total_inserted} documents into '{collection_name}'")
        return total_inserted

    def import_all_collections(
        self,
        input_dir: str | Path,
        file_format: str = "json",
        file_pattern: str = "*",
    ) -> dict[str, int]:
        """Import data from multiple files into MongoDB collections.

        The collection name is derived from the filename (without extension).

        Args:
            input_dir: Directory containing input files
            file_format: File format ('json', 'csv', 'parquet')
            file_pattern: Glob pattern for input files

        Returns:
            Dictionary mapping collection names to number of imported documents
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        # Find files matching pattern and format
        pattern = f"{file_pattern}.{file_format}"
        input_files = list(input_path.glob(pattern))

        if not input_files:
            logger.warning(f"No files found in {input_path} matching {pattern}")
            return {}

        imported_counts = {}

        for input_file in input_files:
            # Derive collection name from filename
            collection_name = input_file.stem

            try:
                count = self.import_collection(
                    collection_name=collection_name,
                    input_file=input_file,
                    file_format=file_format,
                )
                imported_counts[collection_name] = count
            except Exception as e:
                logger.error(f"Failed to import {input_file}: {e}")
                continue

        logger.info(f"Imported {sum(imported_counts.values())} total documents from {len(imported_counts)} files")
        return imported_counts

    def import_documents_by_type(
        self,
        input_dir: str | Path,
        file_format: str = "json",
    ) -> dict[str, int]:
        """Import documents by type with validation.

        Args:
            input_dir: Directory containing input files
            file_format: File format ('json', 'csv', 'parquet')

        Returns:
            Dictionary mapping document types to number of imported documents
        """
        input_path = Path(input_dir)

        document_types = {
            "raw_documents": (RawDocument, "raw_documents"),
            "cleaned_documents": (CleanedDocument, "cleaned_documents"),
            "chunks": (Chunk, "chunks"),
            "embedded_chunks": (EmbeddedChunk, "embedded_chunks"),
        }

        imported_counts = {}

        for doc_type, (doc_class, collection_name) in document_types.items():
            input_file = input_path / f"{doc_type}.{file_format}"

            if not input_file.exists():
                logger.info(f"Skipping {doc_type}: file not found ({input_file})")
                continue

            try:
                count = self.import_collection(
                    collection_name=collection_name,
                    input_file=input_file,
                    file_format=file_format,
                    document_class=doc_class,
                )
                imported_counts[doc_type] = count
            except Exception as e:
                logger.error(f"Failed to import {doc_type}: {e}")
                continue

        return imported_counts


class DataWarehouseManager:
    """Combined manager for export and import operations."""

    def __init__(self, mongodb_manager: MongoDBManager | None = None):
        """Initialize the data warehouse manager.

        Args:
            mongodb_manager: MongoDB manager instance (creates new if None)
        """
        self.exporter = DataWarehouseExporter(mongodb_manager)
        self.importer = DataWarehouseImporter(mongodb_manager)

    def backup(
        self,
        backup_dir: str | Path,
        file_format: str = "json",
        collections: list[str] | None = None,
    ) -> dict[str, Path]:
        """Create a backup of MongoDB collections.

        Args:
            backup_dir: Directory for backup files
            file_format: File format ('json', 'csv', 'parquet')
            collections: Optional list of collections to backup (all if None)

        Returns:
            Dictionary mapping collection names to backup file paths
        """
        logger.info(f"Creating backup to {backup_dir}")
        return self.exporter.export_all_collections(backup_dir, file_format, collections)

    def restore(
        self,
        backup_dir: str | Path,
        file_format: str = "json",
        file_pattern: str = "*",
    ) -> dict[str, int]:
        """Restore MongoDB collections from backup files.

        Args:
            backup_dir: Directory containing backup files
            file_format: File format ('json', 'csv', 'parquet')
            file_pattern: Glob pattern for backup files

        Returns:
            Dictionary mapping collection names to number of restored documents
        """
        logger.info(f"Restoring from backup in {backup_dir}")
        return self.importer.import_all_collections(backup_dir, file_format, file_pattern)

    def migrate(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        source_format: str = "json",
        target_format: str = "parquet",
    ) -> dict[str, Path]:
        """Migrate data between different file formats.

        Args:
            source_dir: Directory containing source files
            target_dir: Directory for target files
            source_format: Source file format
            target_format: Target file format

        Returns:
            Dictionary mapping filenames to migrated file paths
        """
        logger.info(f"Migrating data from {source_format} to {target_format}")

        source_path = Path(source_dir)
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        # Find source files
        source_files = list(source_path.glob(f"*.{source_format}"))

        if not source_files:
            logger.warning(f"No {source_format} files found in {source_path}")
            return {}

        migrated_files = {}

        for source_file in source_files:
            # Read from source format
            if source_format == "json":
                data = DataImporter.import_from_json(source_file)
            elif source_format == "csv":
                data = DataImporter.import_from_csv(source_file)
            elif source_format == "parquet":
                data = DataImporter.import_from_parquet(source_file)
            else:
                logger.warning(f"Unsupported source format: {source_format}")
                continue

            # Handle wrapped data
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            # Write to target format
            target_file = target_path / f"{source_file.stem}.{target_format}"

            try:
                if target_format == "json":
                    output_path = DataExporter.export_to_json(data, target_file, serialize=False)
                elif target_format == "csv":
                    if not isinstance(data, list):
                        data = [data]
                    output_path = DataExporter.export_to_csv(data, target_file)
                elif target_format == "parquet":
                    if not isinstance(data, list):
                        data = [data]
                    output_path = DataExporter.export_to_parquet(data, target_file)
                else:
                    logger.warning(f"Unsupported target format: {target_format}")
                    continue

                migrated_files[source_file.name] = output_path
                logger.info(f"Migrated {source_file.name} to {output_path}")
            except Exception as e:
                logger.error(f"Failed to migrate {source_file.name}: {e}")
                continue

        logger.info(f"Migrated {len(migrated_files)} files from {source_format} to {target_format}")
        return migrated_files

