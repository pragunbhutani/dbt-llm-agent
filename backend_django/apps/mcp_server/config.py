"""Configuration settings for the Ragstar MCP Server."""

import os
from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    """Configuration settings for the MCP server."""

    # Server settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "https://claude.ai",
        "https://*.claude.ai",
        "https://chatgpt.com",
        "https://*.openai.com",
        "http://localhost:3000",  # Local development
        "http://localhost:8000",  # Django backend
    ]

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # Tool settings
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "20"))
    DEFAULT_SEARCH_RESULTS: int = int(os.getenv("DEFAULT_SEARCH_RESULTS", "10"))

    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()
