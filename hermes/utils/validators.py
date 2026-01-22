"""Configuration validators for HermesLLM."""

import sys
from pathlib import Path
from typing import List

import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


def validate_yaml_file(file_path: Path) -> tuple[bool, str | None]:
    """
    Validate a YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, "r") as f:
            yaml.safe_load(f)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_all_configs() -> int:
    """
    Validate all YAML configuration files.

    Returns:
        Exit code (0 for success, 1 for failures)
    """
    console.print("\n[bold cyan]Validating Configuration Files[/bold cyan]\n")

    # Find all YAML files in configs directory
    configs_dir = Path("configs")
    if not configs_dir.exists():
        console.print("[yellow]⚠️  configs/ directory not found[/yellow]")
        return 0

    yaml_files = list(configs_dir.glob("**/*.yaml")) + list(configs_dir.glob("**/*.yml"))

    if not yaml_files:
        console.print("[yellow]⚠️  No YAML files found in configs/[/yellow]")
        return 0

    # Validate each file
    results: List[tuple[Path, bool, str | None]] = []
    for yaml_file in yaml_files:
        is_valid, error = validate_yaml_file(yaml_file)
        results.append((yaml_file, is_valid, error))

    # Display results
    table = Table(title="Configuration Validation Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Error", style="red")

    for file_path, is_valid, error in results:
        status = "✅ Valid" if is_valid else "❌ Invalid"
        error_msg = error or ""
        table.add_row(str(file_path.relative_to(Path.cwd())), status, error_msg)

    console.print(table)

    # Summary
    valid_count = sum(1 for _, is_valid, _ in results if is_valid)
    total_count = len(results)

    console.print(f"\n[bold]Summary:[/bold] {valid_count}/{total_count} files valid\n")

    if valid_count == total_count:
        console.print("[green]✅ All configuration files are valid![/green]")
        return 0
    else:
        console.print("[red]❌ Some configuration files have errors[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(validate_all_configs())
