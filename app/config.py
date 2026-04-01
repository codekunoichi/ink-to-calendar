"""
Configuration and inference client factory.

Reads settings from environment variables (loaded from .env).
Both Ollama (dev) and vLLM (prod) expose an OpenAI-compatible API —
switching between them requires only .env changes, no code changes.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Settings:
    inference_base_url: str = os.getenv("INFERENCE_BASE_URL", "http://localhost:11434/v1")
    inference_model: str = os.getenv("INFERENCE_MODEL", "qwen2.5vl:7b")
    inference_backend: str = os.getenv("INFERENCE_BACKEND", "ollama")

    google_credentials_path: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "./credentials.json")
    google_calendar_id: str = os.getenv("GOOGLE_CALENDAR_ID", "primary")

    app_username: str = os.getenv("APP_USERNAME", "rumpa")
    app_password_hash: str = os.getenv("APP_PASSWORD_HASH", "")
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8080"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_inference_client() -> OpenAI:
    """
    Returns an OpenAI-compatible client pointed at the active backend.

    Ollama uses "ollama" as the API key.
    vLLM accepts any non-empty string.
    """
    settings = get_settings()
    api_key = "ollama" if settings.inference_backend == "ollama" else "vllm"
    return OpenAI(
        base_url=settings.inference_base_url,
        api_key=api_key,
    )
