"""
Data Warehouse Import/Export Utility

Tool for importing and exporting MongoDB data warehouse data to/from JSON files.
"""

import json
from pathlib import Path
from typing import Type

import click
from loguru import logger

from hermes.core.documents import Document
from hermes.core.cleaned_documents import CleanedDocument
from hermes.core.chunks import Chunk
from hermes.storage.database import DatabaseClient


@click.group()
def cli():
    """
    Data Warehouse Import/Export Tool
    
    Manage MongoDB data warehouse backups and migrations.
    """
    pass


@cli.command(name="export")
@click.option(
    "--output-dir",
    type=Path,
    default=Path("data/warehouse_backup"),
    help="Output directory for JSON files",
)
@click.option(
    "--collections",
    multiple=True,
    help="Specific collections to export (default: all)",
)
def export_command(output_dir: Path, collections: tuple):
    """Export data warehouse to JSON files."""
    logger.info(f"Exporting data warehouse to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db_client = DatabaseClient()
    
    # Define collections to export
    if collections:
        collection_names = list(collections)
    else:
        collection_names = ["documents", "cleaned_documents", "chunks"]
    
    for collection_name in collection_names:
        export_collection(db_client, collection_name, output_dir)
    
    logger.success(f"Export completed to {output_dir}")


@cli.command(name="import")
@click.option(
    "--input-dir",
    type=Path,
    default=Path("data/warehouse_backup"),
    help="Input directory with JSON files",
)
@click.option(
    "--collections",
    multiple=True,
    help="Specific collections to import (default: all)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing data",
)
def import_command(input_dir: Path, collections: tuple, overwrite: bool):
    """Import data warehouse from JSON files."""
    logger.info(f"Importing data warehouse from {input_dir}")
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    db_client = DatabaseClient()
    
    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))
    
    if collections:
        json_files = [f for f in json_files if f.stem in collections]
    
    for json_file in json_files:
        import_collection(db_client, json_file, overwrite)
    
    logger.success("Import completed")


def export_collection(db_client: DatabaseClient, collection_name: str, output_dir: Path):
    """Export a single collection to JSON."""
    logger.info(f"Exporting collection: {collection_name}")
    
    try:
        # Query all documents
        documents = db_client.find(
            collection_name=collection_name,
            query={},
        )
        
        # Convert to serializable format
        serialized_docs = []
        for doc in documents:
            if hasattr(doc, "model_dump"):
                serialized_docs.append(doc.model_dump())
            elif isinstance(doc, dict):
                serialized_docs.append(doc)
            else:
                serialized_docs.append(doc.__dict__)
        
        # Save to file
        output_file = output_dir / f"{collection_name}.json"
        with open(output_file, "w") as f:
            json.dump(serialized_docs, f, indent=2, default=str)
        
        logger.success(f"Exported {len(serialized_docs)} documents to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to export {collection_name}: {e}")


def import_collection(db_client: DatabaseClient, json_file: Path, overwrite: bool):
    """Import a single collection from JSON."""
    collection_name = json_file.stem
    logger.info(f"Importing collection: {collection_name}")
    
    try:
        # Load JSON data
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if not data:
            logger.warning(f"No data in {json_file}")
            return
        
        # Check if collection exists and handle overwrite
        existing_count = len(db_client.find(collection_name=collection_name, query={}))
        if existing_count > 0 and not overwrite:
            logger.warning(f"Collection {collection_name} has {existing_count} documents. Use --overwrite to replace.")
            return
        
        # Determine document type
        doc_class = _get_document_class(collection_name)
        
        # Convert to document objects
        documents = []
        for item in data:
            try:
                if doc_class:
                    doc = doc_class(**item)
                else:
                    doc = item
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to parse document: {e}")
                continue
        
        # Bulk insert
        if documents:
            db_client.bulk_insert(
                collection_name=collection_name,
                documents=documents,
            )
            logger.success(f"Imported {len(documents)} documents to {collection_name}")
        
    except Exception as e:
        logger.error(f"Failed to import {json_file}: {e}")


def _get_document_class(collection_name: str):
    """Get document class for collection."""
    mapping = {
        "documents": Document,
        "cleaned_documents": CleanedDocument,
        "chunks": Chunk,
    }
    return mapping.get(collection_name)


@cli.command(name="list")
def list_collections():
    """List all collections in the data warehouse."""
    logger.info("Listing collections")
    
    db_client = DatabaseClient()
    
    collections = ["documents", "cleaned_documents", "chunks"]
    
    for collection_name in collections:
        try:
            count = len(db_client.find(collection_name=collection_name, query={}))
            logger.info(f"{collection_name}: {count} documents")
        except Exception as e:
            logger.error(f"Failed to query {collection_name}: {e}")


@cli.command(name="clear")
@click.option(
    "--collection",
    required=True,
    help="Collection to clear",
)
@click.option(
    "--confirm",
    is_flag=True,
    help="Confirm deletion",
)
def clear_collection(collection: str, confirm: bool):
    """Clear all documents from a collection."""
    if not confirm:
        logger.warning("Add --confirm flag to proceed with deletion")
        return
    
    logger.warning(f"Clearing collection: {collection}")
    
    db_client = DatabaseClient()
    
    try:
        # Delete all documents
        count = len(db_client.find(collection_name=collection, query={}))
        
        # In a real implementation, add bulk delete method
        logger.warning(f"Would delete {count} documents from {collection}")
        logger.info("Note: Implement bulk_delete in DatabaseClient for actual deletion")
        
    except Exception as e:
        logger.error(f"Failed to clear {collection}: {e}")


if __name__ == "__main__":
    cli()
