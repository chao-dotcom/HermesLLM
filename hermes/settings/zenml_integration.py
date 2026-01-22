"""
ZenML Secret Store Integration for Settings Management

This module provides integration with ZenML's secret store for
secure settings management in production environments.
"""

from typing import Dict, Any, Optional
from loguru import logger

from hermes.config import Settings, get_settings


def load_settings_from_zenml() -> Optional[Settings]:
    """
    Load settings from ZenML secret store.
    
    Returns:
        Settings instance loaded from ZenML, or None if not available
    """
    try:
        from zenml.client import Client
        
        logger.info("Loading settings from ZenML secret store...")
        
        client = Client()
        settings_secret = client.get_secret("hermesllm_settings")
        
        # Create Settings instance from secret values
        settings = Settings(**settings_secret.secret_values)
        
        logger.success("Settings loaded from ZenML secret store")
        return settings
        
    except ImportError:
        logger.warning("ZenML not installed. Cannot load settings from secret store.")
        return None
    except (RuntimeError, KeyError) as e:
        logger.warning(f"Failed to load settings from ZenML secret store: {e}")
        logger.info("Falling back to environment variables and .env file")
        return None


def export_settings_to_zenml(settings: Optional[Settings] = None) -> bool:
    """
    Export settings to ZenML secret store.
    
    Args:
        settings: Settings instance to export (uses global if not provided)
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        from zenml.client import Client
        from zenml.exceptions import EntityExistsError
        
        if settings is None:
            settings = get_settings()
        
        logger.info("Exporting settings to ZenML secret store...")
        
        # Convert settings to dictionary
        settings_dict = settings.model_dump()
        
        # Convert all values to strings for ZenML
        secret_values: Dict[str, str] = {}
        for key, value in settings_dict.items():
            if value is not None:
                secret_values[key] = str(value)
        
        client = Client()
        
        try:
            # Try to create new secret
            client.create_secret(
                name="hermesllm_settings",
                values=secret_values,
            )
            logger.success("Settings exported to ZenML secret store")
            return True
            
        except EntityExistsError:
            logger.warning("Secret 'hermesllm_settings' already exists")
            logger.info("Delete it manually with: zenml secret delete hermesllm_settings")
            logger.info("Or update it with: zenml secret update hermesllm_settings")
            return False
            
    except ImportError:
        logger.error("ZenML not installed. Cannot export settings to secret store.")
        return False
    except Exception as e:
        logger.error(f"Failed to export settings to ZenML: {e}")
        return False


def update_settings_in_zenml(
    settings: Optional[Settings] = None,
    updates: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Update existing settings in ZenML secret store.
    
    Args:
        settings: Settings instance to update (uses global if not provided)
        updates: Dictionary of updates to apply
        
    Returns:
        True if update successful, False otherwise
    """
    try:
        from zenml.client import Client
        
        if settings is None:
            settings = get_settings()
        
        logger.info("Updating settings in ZenML secret store...")
        
        # Apply updates if provided
        if updates:
            for key, value in updates.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
                else:
                    logger.warning(f"Unknown setting key: {key}")
        
        # Convert to secret values
        settings_dict = settings.model_dump()
        secret_values: Dict[str, str] = {}
        for key, value in settings_dict.items():
            if value is not None:
                secret_values[key] = str(value)
        
        client = Client()
        
        # Update the secret
        client.update_secret(
            name_id_or_prefix="hermesllm_settings",
            values=secret_values,
        )
        
        logger.success("Settings updated in ZenML secret store")
        return True
        
    except ImportError:
        logger.error("ZenML not installed. Cannot update settings in secret store.")
        return False
    except Exception as e:
        logger.error(f"Failed to update settings in ZenML: {e}")
        return False


def delete_settings_from_zenml() -> bool:
    """
    Delete settings from ZenML secret store.
    
    Returns:
        True if deletion successful, False otherwise
    """
    try:
        from zenml.client import Client
        
        logger.warning("Deleting settings from ZenML secret store...")
        
        client = Client()
        client.delete_secret(name_id_or_prefix="hermesllm_settings")
        
        logger.success("Settings deleted from ZenML secret store")
        return True
        
    except ImportError:
        logger.error("ZenML not installed. Cannot delete settings from secret store.")
        return False
    except Exception as e:
        logger.error(f"Failed to delete settings from ZenML: {e}")
        return False


def list_zenml_secrets() -> Optional[list]:
    """
    List all secrets in ZenML secret store.
    
    Returns:
        List of secret names, or None if not available
    """
    try:
        from zenml.client import Client
        
        client = Client()
        secrets = client.list_secrets()
        
        secret_names = [secret.name for secret in secrets]
        
        logger.info(f"Found {len(secret_names)} secrets in ZenML: {secret_names}")
        return secret_names
        
    except ImportError:
        logger.error("ZenML not installed. Cannot list secrets.")
        return None
    except Exception as e:
        logger.error(f"Failed to list ZenML secrets: {e}")
        return None


def get_setting_from_zenml(key: str) -> Optional[str]:
    """
    Get a specific setting value from ZenML secret store.
    
    Args:
        key: Setting key to retrieve
        
    Returns:
        Setting value as string, or None if not found
    """
    try:
        from zenml.client import Client
        
        client = Client()
        secret = client.get_secret("hermesllm_settings")
        
        value = secret.secret_values.get(key)
        
        if value is not None:
            logger.info(f"Retrieved setting '{key}' from ZenML")
        else:
            logger.warning(f"Setting '{key}' not found in ZenML")
        
        return value
        
    except ImportError:
        logger.error("ZenML not installed. Cannot get setting from secret store.")
        return None
    except Exception as e:
        logger.error(f"Failed to get setting from ZenML: {e}")
        return None


class ZenMLSettingsManager:
    """
    Manager class for ZenML settings operations.
    
    Provides a high-level interface for settings management with ZenML.
    """
    
    def __init__(self):
        """Initialize the ZenML settings manager."""
        try:
            from zenml.client import Client
            self.client = Client()
            self.available = True
        except ImportError:
            logger.warning("ZenML not available. Settings manager will use local settings only.")
            self.available = False
    
    def sync_to_zenml(self, settings: Optional[Settings] = None) -> bool:
        """
        Synchronize local settings to ZenML.
        
        Args:
            settings: Settings to sync (uses global if not provided)
            
        Returns:
            True if successful
        """
        if not self.available:
            logger.error("ZenML not available")
            return False
        
        # Try to export, if exists, update instead
        success = export_settings_to_zenml(settings)
        if not success:
            success = update_settings_in_zenml(settings)
        
        return success
    
    def sync_from_zenml(self) -> Optional[Settings]:
        """
        Synchronize settings from ZenML to local.
        
        Returns:
            Settings instance loaded from ZenML
        """
        if not self.available:
            logger.error("ZenML not available")
            return None
        
        return load_settings_from_zenml()
    
    def ensure_secret_exists(self) -> bool:
        """
        Ensure hermesllm_settings secret exists in ZenML.
        
        Returns:
            True if secret exists or was created
        """
        if not self.available:
            return False
        
        try:
            self.client.get_secret("hermesllm_settings")
            logger.info("hermesllm_settings secret exists in ZenML")
            return True
        except:
            logger.info("hermesllm_settings secret does not exist, creating...")
            return export_settings_to_zenml()
