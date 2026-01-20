"""Configuration management for Atlas LLM."""

from functools import lru_cache
from pathlib import Path

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from zenml.client import Client
    from zenml.exceptions import EntityExistsError
    
    ZENML_AVAILABLE = True
except ImportError:
    ZENML_AVAILABLE = False
    logger.warning("ZenML not available. Secret store functionality disabled.")


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ===== Core Configuration =====
    
    # OpenAI Configuration
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model_id: str = Field("gpt-4o-mini", alias="OPENAI_MODEL_ID")
    
    # HuggingFace Configuration
    huggingface_token: str | None = Field(None, alias="HUGGINGFACE_ACCESS_TOKEN")
    
    # Monitoring & Tracking
    comet_api_key: str | None = Field(None, alias="COMET_API_KEY")
    comet_project: str = Field("atlas", alias="COMET_PROJECT")
    
    # ===== Database Configuration =====
    
    # MongoDB
    database_host: str = Field(
        "mongodb://atlas:atlas@127.0.0.1:27017",
        alias="DATABASE_HOST"
    )
    database_name: str = Field("atlas_db", alias="DATABASE_NAME")
    
    # Qdrant Vector Database
    use_qdrant_cloud: bool = Field(False, alias="USE_QDRANT_CLOUD")
    qdrant_host: str = Field("localhost", alias="QDRANT_DATABASE_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_DATABASE_PORT")
    qdrant_cloud_url: str | None = Field(None, alias="QDRANT_CLOUD_URL")
    qdrant_api_key: str | None = Field(None, alias="QDRANT_APIKEY")
    
    # ===== Cloud Configuration =====
    
    # AWS
    aws_region: str = Field("us-east-1", alias="AWS_REGION")
    aws_access_key: str | None = Field(None, alias="AWS_ACCESS_KEY")
    aws_secret_key: str | None = Field(None, alias="AWS_SECRET_KEY")
    aws_arn_role: str | None = Field(None, alias="AWS_ARN_ROLE")
    
    # AWS SageMaker
    hf_model_id: str = Field(
        "mlabonne/TwinLlama-3.1-8B-DPO",
        alias="HF_MODEL_ID"
    )
    gpu_instance_type: str = Field("ml.g5.2xlarge", alias="GPU_INSTANCE_TYPE")
    sagemaker_num_gpus: int = Field(1, alias="SM_NUM_GPUS")
    max_input_length: int = Field(2048, alias="MAX_INPUT_LENGTH")
    max_total_tokens: int = Field(4096, alias="MAX_TOTAL_TOKENS")
    max_batch_total_tokens: int = Field(4096, alias="MAX_BATCH_TOTAL_TOKENS")
    
    # SageMaker Endpoints
    sagemaker_endpoint_config: str = Field("atlas", alias="SAGEMAKER_ENDPOINT_CONFIG_INFERENCE")
    sagemaker_endpoint_name: str = Field("atlas", alias="SAGEMAKER_ENDPOINT_INFERENCE")
    
    # Inference Parameters
    temperature: float = Field(0.01, alias="TEMPERATURE_INFERENCE")
    top_p: float = Field(0.9, alias="TOP_P_INFERENCE")
    max_new_tokens: int = Field(150, alias="MAX_NEW_TOKENS_INFERENCE")
    
    # ===== Model Configuration =====
    
    # Embeddings & RAG
    embedding_model_id: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        alias="TEXT_EMBEDDING_MODEL_ID"
    )
    reranking_model_id: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-4-v2",
        alias="RERANKING_CROSS_ENCODER_MODEL_ID"
    )
    rag_device: str = Field("cpu", alias="RAG_MODEL_DEVICE")
    
    # ===== Social Media Credentials =====
    
    linkedin_username: str | None = Field(None, alias="LINKEDIN_USERNAME")
    linkedin_password: str | None = Field(None, alias="LINKEDIN_PASSWORD")
    
    # ===== Computed Properties =====
    
    @property
    def openai_max_tokens(self) -> int:
        """Calculate max token window based on model."""
        token_windows = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }
        official_max = token_windows.get(self.openai_model_id, 128000)
        # Use 90% of max to leave buffer
        return int(official_max * 0.90)
    
    @property
    def mongodb_url(self) -> str:
        """Get MongoDB connection URL."""
        return self.database_host
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        if self.use_qdrant_cloud and self.qdrant_cloud_url:
            return self.qdrant_cloud_url
        return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    # ===== ZenML Integration =====
    
    @classmethod
    def from_zenml_secrets(cls) -> "Settings":
        """Load settings from ZenML secret store."""
        if not ZENML_AVAILABLE:
            logger.warning("ZenML not available. Loading from environment.")
            return cls()
        
        try:
            logger.info("Loading settings from ZenML secret store...")
            client = Client()
            secret = client.get_secret("atlas-settings")
            return cls(**secret.secret_values)
        except (RuntimeError, KeyError) as e:
            logger.warning(f"Failed to load from ZenML: {e}. Using environment.")
            return cls()
    
    def export_to_zenml(self) -> None:
        """Export settings to ZenML secret store."""
        if not ZENML_AVAILABLE:
            logger.error("ZenML not available. Cannot export secrets.")
            return
        
        try:
            values = {k: str(v) for k, v in self.model_dump().items()}
            client = Client()
            client.create_secret(name="atlas-settings", values=values)
            logger.info("Settings exported to ZenML secret store.")
        except EntityExistsError:
            logger.warning(
                "Secret 'atlas-settings' exists. "
                "Delete with 'zenml secret delete atlas-settings' first."
            )
        except Exception as e:
            logger.error(f"Failed to export to ZenML: {e}")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    First tries to load from ZenML secret store,
    falls back to environment variables.
    """
    if ZENML_AVAILABLE:
        return Settings.from_zenml_secrets()
    return Settings()


# Global settings instance
settings = get_settings()
