"""Base handler for data processing."""

from abc import ABC, abstractmethod
from typing import Any


class BaseHandler(ABC):
    """Base class for all data processing handlers."""
    
    @abstractmethod
    def handle(self, data: Any) -> Any:
        """
        Process data.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        pass
