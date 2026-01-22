"""Command-line interface for HermesLLM."""

import click
from loguru import logger
from typing import List

from hermes.config import get_settings
from hermes.utils.logging import setup_logging


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """
    HermesLLM - Production-ready LLM system with RAG capabilities.
    
    Use the comprehensive tools for advanced workflows:
    - 'hermes-run' for pipeline orchestration
    - 'hermes-warehouse' for data management
    - 'hermes-rag' for RAG demonstrations
    - 'hermes-serve' for ML inference service
    """
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level)


@cli.command()
@click.argument("urls", nargs=-1, required=True)
@click.option("--user-id", "-u", type=str, required=True, help="User ID")
@click.option("--name", "-n", type=str, required=True, help="User full name")
def collect(urls: tuple, user_id: str, name: str):
    """
    Collect data from URLs.
    
    Example:
        atlas collect https://medium.com/@user/article https://github.com/user/repo -u user123 -n "John Doe"
    """
    from hermes.pipelines.collection import collection_pipeline
    
    logger.info(f"Collecting {len(urls)} URLs for user {user_id}")
    
    try:
        stats = collection_pipeline(
            user_id=user_id,
            full_name=name,
            links=list(urls)
        )
        
        click.echo(f"\n??Collection complete:")
        click.echo(f"  Total: {stats['total']}")
        click.echo(f"  Success: {stats['success']}")
        click.echo(f"  Failed: {stats['failed']}")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        click.echo(f"??Collection failed: {e}", err=True)


@cli.command()
@click.option("--author-id", "-a", type=str, help="Filter by author ID")
@click.option("--limit", "-l", type=int, help="Max documents to process")
@click.option("--chunk-size", type=int, default=500, help="Chunk size")
@click.option("--model", "-m", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
def process(author_id: str | None, limit: int | None, chunk_size: int, model: str):
    """
    Process documents through cleaning, chunking, and embedding.
    
    Example:
        atlas process --author-id user123 --limit 100
    """
    from hermes.pipelines.processing import processing_pipeline
    
    logger.info("Starting processing pipeline")
    
    try:
        stats = processing_pipeline(
            author_id=author_id,
            limit=limit,
            chunk_size=chunk_size,
            embedding_model=model
        )
        
        click.echo(f"\n??Processing complete:")
        click.echo(f"  Documents: {stats.get('total', 0)}")
        click.echo(f"  Vectors stored: {stats.get('stored', 0)}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        click.echo(f"??Processing failed: {e}", err=True)


@cli.command()
@click.option("--data-path", "-d", type=click.Path(exists=True), help="Path to training data")
@click.option("--model", "-m", type=str, default="gpt2", help="Base model")
@click.option("--epochs", "-e", type=int, default=3, help="Training epochs")
@click.option("--output", "-o", type=click.Path(), default="models/fine-tuned", help="Output directory")
def train(data_path: str | None, model: str, epochs: int, output: str):
    """
    Train/fine-tune models.
    
    Example:
        atlas train --data-path data/train.json --model gpt2 --epochs 3
    """
    from hermes.training import LLMTrainer
    
    if not data_path:
        click.echo("Error: --data-path required", err=True)
        return
    
    logger.info(f"Training {model} for {epochs} epochs")
    
    try:
        import json
        from pathlib import Path
        
        # Load data
        with open(data_path) as f:
            data = json.load(f)
        
        # Initialize trainer
        trainer = LLMTrainer(model_name=model, output_dir=output)
        
        # Prepare dataset
        dataset = trainer.prepare_dataset(data)
        
        # Train
        metrics = trainer.train(train_dataset=dataset, num_epochs=epochs)
        
        click.echo(f"\n??Training complete:")
        click.echo(f"  Loss: {metrics['train_loss']:.4f}")
        click.echo(f"  Model saved to: {output}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"??Training failed: {e}", err=True)


@cli.command()
@click.option("--cloud", type=str, default="aws", help="Cloud provider")
def deploy(cloud: str):
    """Deploy model to cloud."""
    logger.info(f"Deploying to {cloud}...")
    click.echo("??Deployment not yet implemented")
    click.echo("Please refer to deployment documentation for manual setup")


@cli.command()
@click.option("--port", "-p", type=int, default=8000, help="Server port")
@click.option("--host", "-h", type=str, default="0.0.0.0", help="Server host")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(port: int, host: str, reload: bool):
    """
    Start API server.
    
    Example:
        atlas serve --port 8000 --reload
    """
    import uvicorn
    from hermes.api import app
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "hermes.api:app",
        host=host,
        port=port,
        reload=reload
    )


@cli.command()
@click.argument("query_text", type=str)
@click.option("--expand", is_flag=True, help="Use query expansion")
@click.option("--k", type=int, default=5, help="Number of results to retrieve")
def query(query_text: str, expand: bool, k: int):
    """
    Query RAG system.
    
    Example:
        atlas query "What are the main features?" --expand --k 5
    """
    from hermes.rag import RAGPipeline
    
    logger.info(f"Querying: {query_text}")
    
    try:
        rag = RAGPipeline()
        answer = rag.query(query=query_text, use_query_expansion=expand)
        
        click.echo(f"\nQuery: {query_text}")
        click.echo(f"\nAnswer:\n{answer}")
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        click.echo(f"??Query failed: {e}", err=True)


@cli.command()
def config():
    """Show current configuration."""
    settings = get_settings()
    click.echo("\n=== Atlas LLM Configuration ===\n")
    click.echo(f"Database: {settings.database_name}")
    click.echo(f"MongoDB: {settings.mongodb_url}")
    click.echo(f"Qdrant: {settings.qdrant_url}")
    click.echo(f"Embedding Model: {settings.embedding_model_id}")
    click.echo(f"Reranking Model: {settings.reranking_model_id}")
    click.echo(f"OpenAI Model: {settings.openai_model_id}")


@cli.command()
def status():
    """Check system status."""
    from hermes.storage.database import MongoDBConnection
    from hermes.storage.vector_store import QdrantStore
    from hermes.core import ArticleDocument, PostDocument, RepositoryDocument, CleanedDocument, Chunk, EmbeddedChunk
    
    click.echo("\n=== Atlas LLM Status ===\n")
    
    # Check MongoDB
    try:
        db = MongoDBConnection()
        db.client.admin.command('ping')
        click.echo("??MongoDB: Connected")
        
        # Count documents
        articles = ArticleDocument.count()
        posts = PostDocument.count()
        repos = RepositoryDocument.count()
        cleaned = CleanedDocument.count()
        chunks = Chunk.count()
        embedded = EmbeddedChunk.count()
        
        click.echo(f"\n  Raw Documents:")
        click.echo(f"    Articles: {articles}")
        click.echo(f"    Posts: {posts}")
        click.echo(f"    Repositories: {repos}")
        click.echo(f"\n  Processed:")
        click.echo(f"    Cleaned: {cleaned}")
        click.echo(f"    Chunks: {chunks}")
        click.echo(f"    Embedded: {embedded}")
        
    except Exception as e:
        click.echo(f"??MongoDB: {e}")
    
    # Check Qdrant
    try:
        vector_store = QdrantStore()
        collections = vector_store.client.get_collections()
        click.echo(f"\n??Qdrant: Connected ({len(collections.collections)} collections)")
    except Exception as e:
        click.echo(f"??Qdrant: {e}")


if __name__ == "__main__":
    cli()
