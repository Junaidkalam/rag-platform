from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = "RAG Platform"
    app_env: str = "development"
    secret_key: str

    # Gemini
    gemini_api_key: str
    gemini_embedding_model: str = "models/text-embedding-004"
    gemini_chat_model: str = "gemini-2.0-flash"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "rag-platform"
    pinecone_environment: str = "us-east-1"

    # Database
    supabase_url: str
    supabase_service_key: str
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Limits
    max_file_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()