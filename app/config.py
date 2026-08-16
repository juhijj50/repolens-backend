"""
Settings loaded from environment variables.
Reads from .env in development, from Render env vars in production.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # API keys
    github_token: str
    groq_api_key: str

    # Database
    database_url: str

    # CORS
    cors_origins: str = "http://localhost:3000"

    # Rate limiting
    max_analyses_per_ip_per_day: int = 10
    max_chat_messages_per_session: int = 20

    # Repo constraints
    max_repo_files: int = 500
    max_repo_size_mb: int = 5
    max_file_size_kb: int = 100

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()
