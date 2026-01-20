"""MongoDB database connection and management."""

from functools import lru_cache

from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure

from atlas.config import get_settings
from atlas.core.base import MongoDocument


class MongoDBConnection:
    """Singleton MongoDB connection manager."""
    
    _instance: MongoClient | None = None
    _database: Database | None = None
    
    def __new__(cls) -> "MongoDBConnection":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self, uri: str | None = None, database_name: str | None = None) -> Database:
        """
        Connect to MongoDB and return database instance.
        
        Args:
            uri: MongoDB connection URI (uses config if not provided)
            database_name: Database name (uses config if not provided)
            
        Returns:
            Database instance
        """
        if self._database is not None:
            return self._database
        
        settings = get_settings()
        uri = uri or settings.mongodb_url
        database_name = database_name or settings.database_name
        
        try:
            client = MongoClient(uri)
            # Test connection
            client.admin.command("ping")
            
            self._database = client[database_name]
            
            # Set database for all MongoDocument subclasses
            MongoDocument.set_database(self._database)
            
            logger.info(f"Connected to MongoDB: {database_name}")
            return self._database
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_database(self) -> Database:
        """Get current database connection."""
        if self._database is None:
            return self.connect()
        return self._database
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self._database is not None:
            self._database.client.close()
            self._database = None
            logger.info("MongoDB connection closed")


@lru_cache()
def get_database() -> Database:
    """Get MongoDB database instance (cached)."""
    connection = MongoDBConnection()
    return connection.get_database()
