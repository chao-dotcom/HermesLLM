"""
Settings Management CLI

Command-line tool for managing HermesLLM settings.
"""

from pathlib import Path
from typing import Optional
import json

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from hermes.config import get_settings, reload_settings, Settings
from hermes.settings import (
    export_settings_to_zenml,
    load_settings_from_zenml,
    update_settings_in_zenml,
    delete_settings_from_zenml,
    list_zenml_secrets,
    ZenMLSettingsManager,
)


console = Console()


@click.group()
def cli():
    """
    Settings Management Tool
    
    Manage HermesLLM configuration settings.
    """
    pass


@cli.command(name="show")
@click.option("--format", type=click.Choice(["table", "json", "env"]), default="table", help="Output format")
@click.option("--show-secrets", is_flag=True, help="Show secret values (API keys, passwords)")
def show_settings(format: str, show_secrets: bool):
    """Display current settings."""
    settings = get_settings()
    
    if format == "table":
        _show_settings_table(settings, show_secrets)
    elif format == "json":
        _show_settings_json(settings, show_secrets)
    elif format == "env":
        _show_settings_env(settings, show_secrets)


def _show_settings_table(settings: Settings, show_secrets: bool):
    """Display settings in table format."""
    table = Table(title="HermesLLM Settings", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="yellow")
    
    settings_dict = settings.model_dump()
    secret_keys = {"openai_api_key", "huggingface_token", "comet_api_key", 
                   "aws_access_key_id", "aws_secret_access_key", "linkedin_password",
                   "qdrant_api_key", "opik_api_key"}
    
    for key, value in sorted(settings_dict.items()):
        # Mask secrets if not showing
        if key in secret_keys and not show_secrets:
            if value:
                value = "***" + str(value)[-4:] if len(str(value)) > 4 else "****"
        
        # Determine source (env var, default, etc.)
        source = "default"
        if value != getattr(Settings(), key, None):
            source = "env/config"
        
        table.add_row(key, str(value), source)
    
    console.print(table)
    
    # Show computed properties
    console.print("\n[bold]Computed Properties:[/bold]")
    console.print(f"qdrant_url: {settings.qdrant_url}")
    console.print(f"openai_max_token_window: {settings.openai_max_token_window}")
    console.print(f"effective_batch_size: {settings.effective_batch_size}")


def _show_settings_json(settings: Settings, show_secrets: bool):
    """Display settings in JSON format."""
    settings_dict = settings.model_dump()
    
    if not show_secrets:
        secret_keys = {"openai_api_key", "huggingface_token", "comet_api_key", 
                       "aws_access_key_id", "aws_secret_access_key", "linkedin_password",
                       "qdrant_api_key", "opik_api_key"}
        for key in secret_keys:
            if key in settings_dict and settings_dict[key]:
                settings_dict[key] = "****"
    
    print(json.dumps(settings_dict, indent=2, default=str))


def _show_settings_env(settings: Settings, show_secrets: bool):
    """Display settings in .env format."""
    settings_dict = settings.model_dump()
    
    secret_keys = {"openai_api_key", "huggingface_token", "comet_api_key", 
                   "aws_access_key_id", "aws_secret_access_key", "linkedin_password",
                   "qdrant_api_key", "opik_api_key"}
    
    for key, value in sorted(settings_dict.items()):
        if value is None:
            continue
        
        # Mask secrets if not showing
        if key in secret_keys and not show_secrets:
            if value:
                value = "****"
        
        env_key = key.upper()
        print(f"{env_key}={value}")


@cli.command(name="export-zenml")
@click.option("--force", is_flag=True, help="Force export even if secret exists")
def export_to_zenml(force: bool):
    """Export settings to ZenML secret store."""
    if force:
        logger.info("Deleting existing secret first...")
        delete_settings_from_zenml()
    
    success = export_settings_to_zenml()
    
    if success:
        console.print("[green]✓[/green] Settings exported to ZenML secret store")
    else:
        console.print("[red]✗[/red] Failed to export settings to ZenML")
        console.print("Try using --force to overwrite existing secret")


@cli.command(name="import-zenml")
def import_from_zenml():
    """Import settings from ZenML secret store."""
    settings = load_settings_from_zenml()
    
    if settings:
        console.print("[green]✓[/green] Settings loaded from ZenML secret store")
        console.print("\nTo persist these settings, export them to .env file:")
        console.print("  hermes-settings export-env --output .env")
    else:
        console.print("[red]✗[/red] Failed to load settings from ZenML")


@cli.command(name="update-zenml")
@click.option("--key", required=True, help="Setting key to update")
@click.option("--value", required=True, help="New value")
def update_in_zenml(key: str, value: str):
    """Update a specific setting in ZenML secret store."""
    updates = {key: value}
    success = update_settings_in_zenml(updates=updates)
    
    if success:
        console.print(f"[green]✓[/green] Updated {key} in ZenML secret store")
    else:
        console.print(f"[red]✗[/red] Failed to update {key} in ZenML")


@cli.command(name="delete-zenml")
@click.option("--confirm", is_flag=True, help="Confirm deletion")
def delete_from_zenml(confirm: bool):
    """Delete settings from ZenML secret store."""
    if not confirm:
        console.print("[yellow]![/yellow] Add --confirm flag to proceed with deletion")
        return
    
    success = delete_settings_from_zenml()
    
    if success:
        console.print("[green]✓[/green] Settings deleted from ZenML secret store")
    else:
        console.print("[red]✗[/red] Failed to delete settings from ZenML")


@cli.command(name="list-zenml")
def list_zenml_secrets_cmd():
    """List all secrets in ZenML secret store."""
    secrets = list_zenml_secrets()
    
    if secrets:
        console.print(f"\n[bold]ZenML Secrets ({len(secrets)}):[/bold]")
        for secret in secrets:
            marker = "✓" if secret == "hermesllm_settings" else "•"
            console.print(f"  {marker} {secret}")
    else:
        console.print("[yellow]![/yellow] No secrets found or ZenML not available")


@cli.command(name="validate")
def validate_settings():
    """Validate current settings."""
    try:
        settings = get_settings()
        console.print("[green]✓[/green] Settings are valid")
        
        # Check for required settings
        warnings = []
        
        if not settings.openai_api_key:
            warnings.append("OpenAI API key not set (required for dataset generation)")
        
        if not settings.huggingface_token:
            warnings.append("HuggingFace token not set (required for model uploads)")
        
        if settings.use_qdrant_cloud and not settings.qdrant_api_key:
            warnings.append("Qdrant Cloud enabled but API key not set")
        
        if settings.use_qdrant_cloud and not settings.qdrant_cloud_url:
            warnings.append("Qdrant Cloud enabled but URL not set")
        
        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  ! {warning}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Settings validation failed: {e}")


@cli.command(name="export-env")
@click.option("--output", type=Path, default=".env.example", help="Output file path")
@click.option("--include-secrets", is_flag=True, help="Include secret values")
def export_to_env_file(output: Path, include_secrets: bool):
    """Export settings to .env file."""
    settings = get_settings()
    settings_dict = settings.model_dump()
    
    secret_keys = {"openai_api_key", "huggingface_token", "comet_api_key", 
                   "aws_access_key_id", "aws_secret_access_key", "linkedin_password",
                   "qdrant_api_key", "opik_api_key"}
    
    with open(output, "w") as f:
        f.write("# HermesLLM Configuration\n")
        f.write("# Generated settings file\n\n")
        
        # Group settings by category
        categories = {
            "API Keys": ["openai_api_key", "openai_model_id", "huggingface_token", 
                        "comet_api_key", "comet_project", "opik_api_key", "opik_workspace"],
            "Database": ["mongodb_url", "database_name", "use_qdrant_cloud", "qdrant_host", 
                        "qdrant_port", "qdrant_cloud_url", "qdrant_api_key"],
            "AWS": ["aws_region", "aws_access_key_id", "aws_secret_access_key", "aws_arn_role"],
            "Models": ["embedding_model_id", "reranking_model_id", "base_model_id"],
            "SageMaker": [k for k in settings_dict.keys() if k.startswith("sagemaker_")],
            "Inference": ["temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty"],
            "RAG": [k for k in settings_dict.keys() if k.startswith("rag_")],
            "Training": ["learning_rate", "num_train_epochs", "per_device_train_batch_size", 
                        "use_lora", "lora_r", "use_4bit"],
            "Processing": ["chunk_size", "chunk_overlap", "min_chunk_size", "max_chunk_size"],
            "Collection": ["linkedin_username", "linkedin_password", "max_articles_per_author"],
        }
        
        for category, keys in categories.items():
            f.write(f"\n# {category}\n")
            for key in keys:
                if key not in settings_dict:
                    continue
                
                value = settings_dict[key]
                if value is None:
                    continue
                
                # Mask secrets if not including
                if key in secret_keys and not include_secrets:
                    value = ""
                
                env_key = key.upper()
                f.write(f"{env_key}={value}\n")
    
    console.print(f"[green]✓[/green] Settings exported to {output}")
    
    if not include_secrets:
        console.print("[yellow]![/yellow] Secret values not included. Use --include-secrets to include them.")


@cli.command(name="init")
@click.option("--env-file", type=Path, default=".env", help="Path to .env file")
def init_settings(env_file: Path):
    """Initialize settings with interactive prompts."""
    console.print("[bold]HermesLLM Settings Initialization[/bold]\n")
    
    settings_dict = {}
    
    # Essential settings
    console.print("[cyan]Essential Settings:[/cyan]")
    
    openai_key = click.prompt("OpenAI API Key (for dataset generation)", default="", hide_input=True)
    if openai_key:
        settings_dict["OPENAI_API_KEY"] = openai_key
    
    hf_token = click.prompt("HuggingFace Token (for model uploads)", default="", hide_input=True)
    if hf_token:
        settings_dict["HUGGINGFACE_TOKEN"] = hf_token
    
    # Database settings
    console.print("\n[cyan]Database Settings:[/cyan]")
    
    mongodb_url = click.prompt("MongoDB URL", default="mongodb://localhost:27017")
    settings_dict["MONGODB_URL"] = mongodb_url
    
    use_qdrant_cloud = click.confirm("Use Qdrant Cloud?", default=False)
    settings_dict["USE_QDRANT_CLOUD"] = str(use_qdrant_cloud)
    
    if use_qdrant_cloud:
        qdrant_url = click.prompt("Qdrant Cloud URL")
        qdrant_key = click.prompt("Qdrant API Key", hide_input=True)
        settings_dict["QDRANT_CLOUD_URL"] = qdrant_url
        settings_dict["QDRANT_API_KEY"] = qdrant_key
    
    # Write to .env file
    with open(env_file, "w") as f:
        f.write("# HermesLLM Configuration\n\n")
        for key, value in settings_dict.items():
            f.write(f"{key}={value}\n")
    
    console.print(f"\n[green]✓[/green] Settings initialized in {env_file}")
    console.print("\nReload settings with: hermes-settings reload")


@cli.command(name="reload")
def reload_settings_cmd():
    """Reload settings from environment."""
    reload_settings()
    console.print("[green]✓[/green] Settings reloaded from environment")


@cli.command(name="sync")
@click.option("--direction", type=click.Choice(["to-zenml", "from-zenml"]), required=True, help="Sync direction")
def sync_settings(direction: str):
    """Synchronize settings between local and ZenML."""
    manager = ZenMLSettingsManager()
    
    if direction == "to-zenml":
        success = manager.sync_to_zenml()
        if success:
            console.print("[green]✓[/green] Settings synchronized to ZenML")
        else:
            console.print("[red]✗[/red] Failed to sync settings to ZenML")
    else:
        settings = manager.sync_from_zenml()
        if settings:
            console.print("[green]✓[/green] Settings synchronized from ZenML")
        else:
            console.print("[red]✗[/red] Failed to sync settings from ZenML")


if __name__ == "__main__":
    cli()
