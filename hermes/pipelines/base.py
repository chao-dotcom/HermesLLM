"""Base pipeline configuration for ZenML."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from zenml import pipeline
from loguru import logger


class BasePipeline(ABC):
    """Base class for all ZenML pipelines."""
    
    def __init__(self, name: str, **kwargs) -> None:
        """
        Initialize pipeline.
        
        Args:
            name: Pipeline name
            **kwargs: Additional pipeline configuration
        """
        self.name = name
        self.config = kwargs
        logger.info(f"Initialized pipeline: {name}")
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Execute pipeline.
        
        Args:
            **kwargs: Pipeline parameters
            
        Returns:
            Pipeline result
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return {
            "name": self.name,
            **self.config
        }
