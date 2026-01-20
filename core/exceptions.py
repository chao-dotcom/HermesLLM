"""Custom exceptions for Atlas LLM."""


class AtlasException(Exception):
    """Base exception for Atlas LLM."""
    pass


class ImproperlyConfigured(AtlasException):
    """Raised when configuration is invalid."""
    pass


class CrawlerException(AtlasException):
    """Raised when crawler encounters an error."""
    pass


class ProcessingException(AtlasException):
    """Raised during data processing."""
    pass


class StorageException(AtlasException):
    """Raised during storage operations."""
    pass


class ModelException(AtlasException):
    """Raised during model operations."""
    pass


class ValidationException(AtlasException):
    """Raised when validation fails."""
    pass
