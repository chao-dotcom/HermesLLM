"""
Settings Management Examples

This module demonstrates various ways to use the HermesLLM settings system.
"""

from pathlib import Path
from hermes.config import get_settings, reload_settings
from hermes.settings import (
    load_settings_from_zenml,
    export_settings_to_zenml,
    update_settings_in_zenml,
    ZenMLSettingsManager,
)


def example_basic_usage():
    """Example 1: Basic settings usage."""
    print("\n=== Example 1: Basic Settings Usage ===\n")
    
    # Get global settings instance
    settings = get_settings()
    
    # Access settings
    print(f"OpenAI Model: {settings.openai_model_id}")
    print(f"Database: {settings.database_name}")
    print(f"Temperature: {settings.temperature}")
    print(f"Use LoRA: {settings.use_lora}")
    print(f"Embedding Model: {settings.embedding_model_id}")


def example_computed_properties():
    """Example 2: Using computed properties."""
    print("\n=== Example 2: Computed Properties ===\n")
    
    settings = get_settings()
    
    # Qdrant URL (auto-selects cloud or local)
    print(f"Qdrant URL: {settings.qdrant_url}")
    
    # OpenAI max token window (90% of model limit)
    print(f"OpenAI Max Tokens: {settings.openai_max_token_window}")
    
    # Effective batch size with gradient accumulation
    print(f"Batch Size: {settings.per_device_train_batch_size}")
    print(f"Grad Accum Steps: {settings.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {settings.effective_batch_size}")


def example_environment_override():
    """Example 3: Overriding with environment variables."""
    print("\n=== Example 3: Environment Variable Override ===\n")
    
    import os
    
    # Set environment variables
    os.environ["TEMPERATURE"] = "0.9"
    os.environ["MAX_NEW_TOKENS"] = "1024"
    os.environ["USE_LORA"] = "false"
    
    # Reload settings to pick up changes
    settings = reload_settings()
    
    print(f"Temperature: {settings.temperature}")
    print(f"Max New Tokens: {settings.max_new_tokens}")
    print(f"Use LoRA: {settings.use_lora}")
    
    # Clean up
    del os.environ["TEMPERATURE"]
    del os.environ["MAX_NEW_TOKENS"]
    del os.environ["USE_LORA"]


def example_validation():
    """Example 4: Settings validation."""
    print("\n=== Example 4: Settings Validation ===\n")
    
    settings = get_settings()
    
    # Check required fields
    required_fields = [
        "openai_api_key",
        "huggingface_token",
        "comet_api_key",
    ]
    
    missing = []
    for field in required_fields:
        value = getattr(settings, field, None)
        if not value:
            missing.append(field)
    
    if missing:
        print(f"⚠️  Missing required settings: {', '.join(missing)}")
    else:
        print("✅ All required settings configured")
    
    # Validate ranges
    if not (0.0 <= settings.temperature <= 2.0):
        print(f"⚠️  Temperature out of range: {settings.temperature}")
    else:
        print(f"✅ Temperature valid: {settings.temperature}")
    
    if not (0.0 <= settings.top_p <= 1.0):
        print(f"⚠️  Top P out of range: {settings.top_p}")
    else:
        print(f"✅ Top P valid: {settings.top_p}")


def example_zenml_export():
    """Example 5: Export settings to ZenML."""
    print("\n=== Example 5: Export to ZenML ===\n")
    
    try:
        # Export current settings to ZenML
        success = export_settings_to_zenml()
        
        if success:
            print("✅ Settings exported to ZenML secret store")
        else:
            print("❌ Failed to export settings")
    except Exception as e:
        print(f"⚠️  ZenML not available: {e}")


def example_zenml_import():
    """Example 6: Import settings from ZenML."""
    print("\n=== Example 6: Import from ZenML ===\n")
    
    try:
        # Load settings from ZenML
        settings = load_settings_from_zenml()
        
        if settings:
            print("✅ Settings loaded from ZenML")
            print(f"   Model: {settings.get('openai_model_id')}")
            print(f"   Database: {settings.get('database_name')}")
        else:
            print("⚠️  No settings found in ZenML, using environment variables")
    except Exception as e:
        print(f"⚠️  ZenML not available: {e}")


def example_zenml_update():
    """Example 7: Update specific settings in ZenML."""
    print("\n=== Example 7: Update ZenML Settings ===\n")
    
    try:
        # Update specific settings
        updates = {
            "temperature": 0.8,
            "max_new_tokens": 1024,
            "top_p": 0.95,
        }
        
        success = update_settings_in_zenml(updates=updates)
        
        if success:
            print("✅ Settings updated in ZenML")
            for key, value in updates.items():
                print(f"   {key}: {value}")
        else:
            print("❌ Failed to update settings")
    except Exception as e:
        print(f"⚠️  ZenML not available: {e}")


def example_zenml_manager():
    """Example 8: Using ZenML Settings Manager."""
    print("\n=== Example 8: ZenML Settings Manager ===\n")
    
    try:
        manager = ZenMLSettingsManager()
        
        # Sync current settings to ZenML
        print("Syncing to ZenML...")
        manager.sync_to_zenml()
        print("✅ Synced to ZenML")
        
        # Sync from ZenML
        print("\nSyncing from ZenML...")
        settings = manager.sync_from_zenml()
        if settings:
            print("✅ Synced from ZenML")
            print(f"   Loaded {len(settings)} settings")
        
        # Ensure secret exists
        print("\nEnsuring secret exists...")
        manager.ensure_secret_exists()
        print("✅ Secret exists")
        
    except Exception as e:
        print(f"⚠️  ZenML not available: {e}")


def example_model_dump():
    """Example 9: Export settings to dictionary."""
    print("\n=== Example 9: Export to Dictionary ===\n")
    
    settings = get_settings()
    
    # Export all settings
    all_settings = settings.model_dump()
    print(f"Total settings: {len(all_settings)}")
    
    # Export only non-default values
    non_defaults = {
        key: value
        for key, value in all_settings.items()
        if value is not None and value != ""
    }
    print(f"Configured settings: {len(non_defaults)}")
    
    # Export specific category
    model_settings = {
        key: value
        for key, value in all_settings.items()
        if "model" in key.lower()
    }
    print(f"\nModel-related settings: {len(model_settings)}")
    for key, value in model_settings.items():
        print(f"  {key}: {value}")


def example_env_file_export():
    """Example 10: Export to .env file."""
    print("\n=== Example 10: Export to .env File ===\n")
    
    settings = get_settings()
    
    # Export to .env.example (without secrets)
    output_file = Path(".env.example")
    
    with output_file.open("w") as f:
        f.write("# HermesLLM Configuration\n\n")
        
        # API Keys
        f.write("# API Keys\n")
        f.write(f"OPENAI_MODEL_ID={settings.openai_model_id}\n")
        f.write("OPENAI_API_KEY=your-api-key-here\n")
        f.write("HUGGINGFACE_TOKEN=your-token-here\n")
        f.write("COMET_API_KEY=your-api-key-here\n\n")
        
        # Database
        f.write("# Database\n")
        f.write(f"MONGODB_URL={settings.mongodb_url}\n")
        f.write(f"DATABASE_NAME={settings.database_name}\n")
        f.write(f"USE_QDRANT_CLOUD={settings.use_qdrant_cloud}\n\n")
        
        # Models
        f.write("# Models\n")
        f.write(f"EMBEDDING_MODEL_ID={settings.embedding_model_id}\n")
        f.write(f"RERANKING_MODEL_ID={settings.reranking_model_id}\n")
        f.write(f"BASE_MODEL_ID={settings.base_model_id}\n\n")
        
        # Inference
        f.write("# Inference\n")
        f.write(f"TEMPERATURE={settings.temperature}\n")
        f.write(f"TOP_P={settings.top_p}\n")
        f.write(f"MAX_NEW_TOKENS={settings.max_new_tokens}\n\n")
        
        # Training
        f.write("# Training\n")
        f.write(f"LEARNING_RATE={settings.learning_rate}\n")
        f.write(f"NUM_TRAIN_EPOCHS={settings.num_train_epochs}\n")
        f.write(f"USE_LORA={settings.use_lora}\n")
        f.write(f"USE_4BIT={settings.use_4bit}\n")
    
    print(f"✅ Exported to {output_file}")


def example_production_pattern():
    """Example 11: Production deployment pattern."""
    print("\n=== Example 11: Production Pattern ===\n")
    
    import os
    
    # Production pattern: Try ZenML first, fall back to env vars
    settings = None
    
    # Check if in production
    is_production = os.getenv("ENVIRONMENT") == "production"
    
    if is_production:
        print("Production environment detected")
        
        try:
            # Try loading from ZenML
            settings_dict = load_settings_from_zenml()
            if settings_dict:
                print("✅ Loaded settings from ZenML")
                settings = get_settings()
            else:
                print("⚠️  No ZenML settings, using environment variables")
                settings = get_settings()
        except Exception as e:
            print(f"⚠️  ZenML error: {e}")
            print("Falling back to environment variables")
            settings = get_settings()
    else:
        print("Development environment")
        print("Using .env file and environment variables")
        settings = get_settings()
    
    print(f"\nConfiguration loaded:")
    print(f"  Database: {settings.database_name}")
    print(f"  Model: {settings.base_model_id}")
    print(f"  Qdrant: {settings.qdrant_url}")


def main():
    """Run all examples."""
    examples = [
        example_basic_usage,
        example_computed_properties,
        example_environment_override,
        example_validation,
        example_zenml_export,
        example_zenml_import,
        example_zenml_update,
        example_zenml_manager,
        example_model_dump,
        example_env_file_export,
        example_production_pattern,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
