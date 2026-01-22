"""
Settings and Configuration Management for HermesLLM

This module provides comprehensive settings management with:
- Environment variable support (.env files)
- ZenML secret store integration
- Pydantic validation
- Default values for all settings
- Export/import functionality
"""

from typing import Optional
from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    Settings are loaded in the following order (later overrides earlier):
    1. Default values defined here
    2. Environment variables
    3. .env file
    4. ZenML secret store (if available)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )
    
    # ========================================================================
    # API Keys & Authentication (Required for full functionality)
    # ========================================================================
    
    # OpenAI API (for dataset generation)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for dataset generation"
    )
    openai_model_id: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model ID for dataset generation"
    )
    
    # HuggingFace API (for model uploads)
    huggingface_token: Optional[str] = Field(
        default=None,
        description="HuggingFace access token for model/dataset uploads"
    )
    
    # Comet ML (for experiment tracking)
    comet_api_key: Optional[str] = Field(
        default=None,
        description="Comet ML API key for experiment tracking"
    )
    comet_project: str = Field(
        default="hermesllm",
        description="Comet ML project name"
    )
    
    # Opik (for monitoring)
    opik_api_key: Optional[str] = Field(
        default=None,
        description="Opik API key for monitoring"
    )
    opik_workspace: Optional[str] = Field(
        default=None,
        description="Opik workspace name"
    )
    
    # ========================================================================
    # Database Configuration
    # ========================================================================
    
    # MongoDB
    mongodb_url: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URL"
    )
    database_name: str = Field(
        default="hermesllm",
        description="MongoDB database name"
    )
    
    # Qdrant Vector Database
    use_qdrant_cloud: bool = Field(
        default=False,
        description="Use Qdrant Cloud instead of local instance"
    )
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant host (for local instance)"
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant port (for local instance)"
    )
    qdrant_cloud_url: Optional[str] = Field(
        default=None,
        description="Qdrant Cloud URL"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (for cloud)"
    )
    
    # ========================================================================
    # AWS Configuration (for SageMaker deployment)
    # ========================================================================
    
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region"
    )
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key"
    )
    aws_arn_role: Optional[str] = Field(
        default=None,
        description="AWS ARN role for SageMaker"
    )
    
    # ========================================================================
    # Model Configuration
    # ========================================================================
    
    # Embedding Models
    embedding_model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    embedding_model_device: str = Field(
        default="cpu",
        description="Device for embedding model (cpu/cuda)"
    )
    embedding_model_max_length: int = Field(
        default=512,
        description="Maximum sequence length for embeddings"
    )
    
    # Reranking Models
    reranking_model_id: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-4-v2",
        description="Cross-encoder model for reranking"
    )
    reranking_model_device: str = Field(
        default="cpu",
        description="Device for reranking model (cpu/cuda)"
    )
    
    # Base Models (for training/inference)
    base_model_id: str = Field(
        default="meta-llama/Llama-2-7b-hf",
        description="Base model for fine-tuning"
    )
    
    # ========================================================================
    # SageMaker Configuration
    # ========================================================================
    
    # SageMaker Instance Configuration
    sagemaker_instance_type: str = Field(
        default="ml.g5.2xlarge",
        description="SageMaker instance type"
    )
    sagemaker_instance_count: int = Field(
        default=1,
        description="Number of SageMaker instances"
    )
    sagemaker_gpu_count: int = Field(
        default=1,
        description="Number of GPUs per instance"
    )
    sagemaker_cpu_count: int = Field(
        default=2,
        description="Number of CPU cores"
    )
    
    # SageMaker Endpoint Configuration
    sagemaker_endpoint_name: str = Field(
        default="hermesllm-endpoint",
        description="SageMaker endpoint name"
    )
    sagemaker_endpoint_config_name: str = Field(
        default="hermesllm-endpoint-config",
        description="SageMaker endpoint configuration name"
    )
    
    # SageMaker Inference Configuration
    sagemaker_max_input_length: int = Field(
        default=2048,
        description="Maximum input length for SageMaker inference"
    )
    sagemaker_max_total_tokens: int = Field(
        default=4096,
        description="Maximum total tokens for SageMaker"
    )
    sagemaker_max_batch_total_tokens: int = Field(
        default=4096,
        description="Maximum batch total tokens for SageMaker"
    )
    
    # ========================================================================
    # Inference Configuration
    # ========================================================================
    
    # Generation Parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        description="Top-k sampling parameter"
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        description="Maximum number of tokens to generate"
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        description="Repetition penalty"
    )
    
    # ========================================================================
    # RAG Configuration
    # ========================================================================
    
    # Retrieval Parameters
    rag_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of documents to retrieve for RAG"
    )
    rag_use_reranking: bool = Field(
        default=True,
        description="Use cross-encoder reranking in RAG"
    )
    rag_use_query_expansion: bool = Field(
        default=False,
        description="Use multi-query expansion in RAG"
    )
    rag_num_expanded_queries: int = Field(
        default=3,
        ge=1,
        description="Number of expanded queries to generate"
    )
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    
    # Training Hyperparameters
    learning_rate: float = Field(
        default=3e-4,
        gt=0.0,
        description="Learning rate for training"
    )
    num_train_epochs: int = Field(
        default=3,
        ge=1,
        description="Number of training epochs"
    )
    per_device_train_batch_size: int = Field(
        default=2,
        ge=1,
        description="Training batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        description="Gradient accumulation steps"
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Number of warmup steps"
    )
    
    # LoRA Configuration
    use_lora: bool = Field(
        default=True,
        description="Use LoRA for efficient fine-tuning"
    )
    lora_r: int = Field(
        default=16,
        ge=1,
        description="LoRA rank"
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        description="LoRA alpha"
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="LoRA dropout"
    )
    
    # Quantization
    use_4bit: bool = Field(
        default=True,
        description="Use 4-bit quantization"
    )
    use_8bit: bool = Field(
        default=False,
        description="Use 8-bit quantization"
    )
    
    # ========================================================================
    # Processing Configuration
    # ========================================================================
    
    # Chunking Parameters
    chunk_size: int = Field(
        default=512,
        ge=1,
        description="Default chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between chunks in tokens"
    )
    min_chunk_size: int = Field(
        default=100,
        ge=1,
        description="Minimum chunk size in tokens"
    )
    max_chunk_size: int = Field(
        default=1024,
        ge=1,
        description="Maximum chunk size in tokens"
    )
    
    # ========================================================================
    # Data Collection Configuration
    # ========================================================================
    
    # LinkedIn Credentials (for LinkedIn collector)
    linkedin_username: Optional[str] = Field(
        default=None,
        description="LinkedIn username"
    )
    linkedin_password: Optional[str] = Field(
        default=None,
        description="LinkedIn password"
    )
    
    # Collection Limits
    max_articles_per_author: int = Field(
        default=100,
        ge=1,
        description="Maximum articles to collect per author"
    )
    max_repos_per_author: int = Field(
        default=50,
        ge=1,
        description="Maximum GitHub repos to collect per author"
    )
    
    # ========================================================================
    # ZenML Configuration
    # ========================================================================
    
    zenml_server_url: Optional[str] = Field(
        default=None,
        description="ZenML server URL (for remote server)"
    )
    zenml_store_type: str = Field(
        default="local",
        description="ZenML store type (local/sql/rest)"
    )
    
    # ========================================================================
    # Computed Properties
    # ========================================================================
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant URL based on configuration."""
        if self.use_qdrant_cloud and self.qdrant_cloud_url:
            return self.qdrant_cloud_url
        return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    @property
    def openai_max_token_window(self) -> int:
        """
        Calculate maximum token window for OpenAI model (90% of official limit).
        
        Returns:
            Maximum token window (90% of official limit)
        """
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.openai_model_id, 128000)
        
        # Use 90% of max to leave room for system prompts and safety margin
        max_token_window = int(official_max_token_window * 0.90)
        
        return max_token_window
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
    
    # ========================================================================
    # Validation
    # ========================================================================
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        """Validate top_p is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v
    
    @field_validator("mongodb_url")
    @classmethod
    def validate_mongodb_url(cls, v: str) -> str:
        """Ensure MongoDB URL has correct protocol."""
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            logger.warning(f"MongoDB URL should start with 'mongodb://' or 'mongodb+srv://': {v}")
        return v


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(force_reload: bool = False) -> Settings:
    """
    Get global settings instance (singleton pattern).
    
    Args:
        force_reload: Force reload settings from environment
        
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None or force_reload:
        _settings = Settings()
        logger.info("Settings loaded successfully")
    
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables.
    
    Returns:
        Fresh Settings instance
    """
    return get_settings(force_reload=True)
