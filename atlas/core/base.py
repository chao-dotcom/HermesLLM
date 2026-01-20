"""Base classes for domain models."""

import uuid
from abc import ABC
from typing import Any, Generic, Type, TypeVar

from loguru import logger
from pydantic import UUID4, BaseModel, Field
from pymongo import errors
from pymongo.database import Database

T = TypeVar("T", bound="MongoDocument")


class MongoDocument(BaseModel, Generic[T], ABC):
    """
    Base class for MongoDB documents with ORM-like functionality.
    
    Provides CRUD operations and automatic UUID handling.
    """
    
    id: UUID4 = Field(default_factory=uuid.uuid4)
    
    # Class-level database reference (injected at runtime)
    _database: Database | None = None
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    # ===== Serialization =====
    
    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        """Convert MongoDB document to Python model."""
        if not data:
            raise ValueError("Cannot create instance from empty data")
        
        # Convert _id to id
        mongo_id = data.pop("_id", None)
        if mongo_id:
            data["id"] = mongo_id
        
        return cls(**data)
    
    def to_mongo(self, **kwargs) -> dict:
        """Convert Python model to MongoDB document."""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)
        
        data = self.model_dump(
            exclude_unset=exclude_unset,
            by_alias=by_alias,
            **kwargs
        )
        
        # Convert id to _id
        if "_id" not in data and "id" in data:
            data["_id"] = str(data.pop("id"))
        
        # Convert any UUIDs to strings
        for key, value in data.items():
            if isinstance(value, uuid.UUID):
                data[key] = str(value)
        
        return data
    
    def model_dump(self, **kwargs) -> dict:
        """Override to handle UUID serialization."""
        data = super().model_dump(**kwargs)
        
        for key, value in data.items():
            if isinstance(value, uuid.UUID):
                data[key] = str(value)
        
        return data
    
    # ===== CRUD Operations =====
    
    def save(self: T, **kwargs) -> T | None:
        """Save document to MongoDB."""
        try:
            collection = self._get_collection()
            collection.insert_one(self.to_mongo(**kwargs))
            logger.debug(f"Saved {self.__class__.__name__} with id {self.id}")
            return self
        except errors.WriteError as e:
            logger.error(f"Failed to save {self.__class__.__name__}: {e}")
            return None
    
    @classmethod
    def find_by_id(cls: Type[T], doc_id: UUID4 | str) -> T | None:
        """Find document by ID."""
        try:
            collection = cls._get_collection()
            data = collection.find_one({"_id": str(doc_id)})
            return cls.from_mongo(data) if data else None
        except errors.OperationFailure as e:
            logger.error(f"Failed to find {cls.__name__}: {e}")
            return None
    
    @classmethod
    def find_one(cls: Type[T], **filters) -> T | None:
        """Find single document by filters."""
        try:
            collection = cls._get_collection()
            data = collection.find_one(filters)
            return cls.from_mongo(data) if data else None
        except errors.OperationFailure as e:
            logger.error(f"Failed to find {cls.__name__}: {e}")
            return None
    
    @classmethod
    def find_many(cls: Type[T], **filters) -> list[T]:
        """Find multiple documents by filters."""
        try:
            collection = cls._get_collection()
            cursor = collection.find(filters)
            return [cls.from_mongo(doc) for doc in cursor]
        except errors.OperationFailure as e:
            logger.error(f"Failed to find {cls.__name__} documents: {e}")
            return []
    
    @classmethod
    def get_or_create(cls: Type[T], **filters) -> T:
        """Get existing document or create new one."""
        existing = cls.find_one(**filters)
        if existing:
            return existing
        
        new_doc = cls(**filters)
        new_doc.save()
        return new_doc
    
    @classmethod
    def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool:
        """Insert multiple documents at once."""
        try:
            collection = cls._get_collection()
            collection.insert_many([doc.to_mongo(**kwargs) for doc in documents])
            logger.info(f"Bulk inserted {len(documents)} {cls.__name__} documents")
            return True
        except (errors.WriteError, errors.BulkWriteError) as e:
            logger.error(f"Failed to bulk insert {cls.__name__}: {e}")
            return False
    
    def update(self: T, **updates) -> bool:
        """Update document fields."""
        try:
            collection = self._get_collection()
            result = collection.update_one(
                {"_id": str(self.id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except errors.OperationFailure as e:
            logger.error(f"Failed to update {self.__class__.__name__}: {e}")
            return False
    
    def delete(self) -> bool:
        """Delete this document."""
        try:
            collection = self._get_collection()
            result = collection.delete_one({"_id": str(self.id)})
            return result.deleted_count > 0
        except errors.OperationFailure as e:
            logger.error(f"Failed to delete {self.__class__.__name__}: {e}")
            return False
    
    @classmethod
    def count(cls: Type[T], **filters) -> int:
        """Count documents matching filters."""
        try:
            collection = cls._get_collection()
            return collection.count_documents(filters)
        except errors.OperationFailure as e:
            logger.error(f"Failed to count {cls.__name__}: {e}")
            return 0
    
    # ===== Collection Management =====
    
    @classmethod
    def _get_collection(cls):
        """Get MongoDB collection for this document type."""
        if cls._database is None:
            raise RuntimeError(
                f"{cls.__name__} database not initialized. "
                "Call MongoDocument.set_database(db) first."
            )
        
        collection_name = cls.get_collection_name()
        return cls._database[collection_name]
    
    @classmethod
    def get_collection_name(cls) -> str:
        """Get collection name from Settings."""
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise AttributeError(
                f"{cls.__name__} must define a Settings class with 'name' attribute"
            )
        return cls.Settings.name
    
    @classmethod
    def set_database(cls, database: Database) -> None:
        """Set database for all document types."""
        cls._database = database


class VectorDocument(BaseModel, ABC):
    """Base class for vector database documents."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: list[float] = Field(description="Embedding vector")
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    def to_qdrant_point(self) -> dict:
        """Convert to Qdrant point format."""
        return {
            "id": self.id,
            "vector": self.vector,
            "payload": self.metadata
        }
