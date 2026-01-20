"""Dispatcher for routing documents to appropriate cleaners."""

from loguru import logger

from hermes.core import ArticleDocument, PostDocument, RepositoryDocument
from hermes.core.base import MongoDocument
from hermes.processing.base import BaseHandler


class CleaningDispatcher:
    """Dispatcher for routing documents to appropriate cleaners."""
    
    def __init__(self) -> None:
        self._cleaners: dict[type[MongoDocument], type[BaseHandler]] = {}
    
    @classmethod
    def build(cls) -> "CleaningDispatcher":
        """Build dispatcher with default cleaners registered."""
        dispatcher = cls()
        return dispatcher
    
    def register_article(self) -> "CleaningDispatcher":
        """Register article cleaner."""
        from hermes.processing.cleaners import ArticleCleaner
        self.register(ArticleDocument, ArticleCleaner)
        return self
    
    def register_post(self) -> "CleaningDispatcher":
        """Register post cleaner."""
        from hermes.processing.cleaners import PostCleaner
        self.register(PostDocument, PostCleaner)
        return self
    
    def register_repository(self) -> "CleaningDispatcher":
        """Register repository cleaner."""
        from hermes.processing.cleaners import RepositoryCleaner
        self.register(RepositoryDocument, RepositoryCleaner)
        return self
    
    def register(self, document_type: type[MongoDocument], cleaner: type[BaseHandler]) -> None:
        """
        Register a cleaner for a document type.
        
        Args:
            document_type: Document class
            cleaner: Cleaner class
        """
        self._cleaners[document_type] = cleaner
        logger.info(f"Registered {cleaner.__name__} for {document_type.__name__}")
    
    def get_cleaner(self, document: MongoDocument) -> BaseHandler:
        """
        Get appropriate cleaner for document.
        
        Args:
            document: Document to clean
            
        Returns:
            Cleaner instance
        """
        document_type = type(document)
        
        if document_type in self._cleaners:
            cleaner_class = self._cleaners[document_type]
            logger.debug(f"Using {cleaner_class.__name__} for {document_type.__name__}")
            return cleaner_class()
        
        raise ValueError(f"No cleaner registered for {document_type.__name__}")
