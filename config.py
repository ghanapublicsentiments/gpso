"""Environment configuration loader for GPSO pipeline.

This module handles:
- Loading environment variables from .env file
- Configuring BigQuery settings
- Managing LLM provider configurations
- Creating OpenAI-compatible clients for various LLM providers
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


# Load .env file from project root (same directory as this file)
project_root = Path(__file__).parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
elif not any(os.getenv(f"{provider}_API_KEY") for provider in ["OPENAI", "ANTHROPIC", "GOOGLE", "GROK", "NVIDIA"]):
    print(f"⚠️  Warning: .env file not found at {env_path}")
    print("   Copy .env.sample to .env and fill in your API keys")


# BigQuery Configuration
BIGQUERY_DATASET: str = os.getenv("BIGQUERY_DATASET", "gpso_main")
BIGQUERY_CREDENTIALS_PATH: Path = project_root / "bigquery-credentials.json"
BIGQUERY_IS_PROD: bool = os.getenv("BIGQUERY_IS_PROD", "true").lower() in ("true", "1", "yes")


# LLM Provider Configuration
PROVIDER_BASE_URL: dict[str, str] = {
    "ANTHROPIC": "https://api.anthropic.com/v1/",
    "GOOGLE": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "GROK": "https://api.x.ai/v1/",
    "NVIDIA": "https://integrate.api.nvidia.com/v1",
    "OPENAI": "https://api.openai.com/v1/"
}

MODEL_PROVIDER_MAP: dict[str, str] = {

    # Google Gemini models
    "gemini-3-flash-preview": "GOOGLE",
    "gemini-3-pro-preview": "GOOGLE",

    # OpenAI models
    "gpt-5-nano": "OPENAI",
    "gpt-5-mini": "OPENAI",
    "gpt-5.1": "OPENAI",
    "gpt-5.2": "OPENAI",

    # Anthropic models
    "claude-haiku-4-5": "ANTHROPIC",
    "claude-opus-4-5": "ANTHROPIC",
    "claude-sonnet-4-5": "ANTHROPIC",

    # Grok models
    "grok-4-1-fast-reasoning": "GROK",
    "grok-4-1-fast-non-reasoning": "GROK",

    # NVIDIA open-source models
    "openai/gpt-oss-20b": "NVIDIA"
}

# Default model for Streamlit app
DEFAULT_MODEL: str = "gemini-3-flash-preview"



def get_client(model: str, api_key: Optional[str] = None) -> OpenAI:
    """Get OpenAI-compatible client for the specified model.

    Args:
        model: Model name from MODEL_PROVIDER_MAP.
        api_key: Optional user-provided API key. If None, retrieves from environment.

    Returns:
        OpenAI: Configured OpenAI client instance.

    Raises:
        ValueError: If model is not found in MODEL_PROVIDER_MAP.
        ValueError: If API key is required but not provided or found in environment.
    """
    provider = MODEL_PROVIDER_MAP.get(model)
    if not provider:
        raise ValueError(
            f"""
            Model '{model}' not found in MODEL_PROVIDER_MAP.
            Available models: {', '.join(MODEL_PROVIDER_MAP.keys())}
            """.strip()
        )

    base_url = PROVIDER_BASE_URL.get(provider)
    if not base_url:
        raise ValueError(f"Provider '{provider}' not found in PROVIDER_BASE_URL.")

    if not api_key:
        api_key = os.getenv(f"{provider}_API_KEY")
        if not api_key:
            error_msg = f"""
            API key for {provider} not found.
            
            To fix this:
            1. Create a .env file in the project root (copy from .env.sample)
            2. Add your API key: {provider}_API_KEY=your-api-key-here
            3. Restart the application
            
            If you're using Streamlit Cloud:
            - Add the key to Streamlit secrets on Streamlit Cloud
            
            If providing the key directly:
            - Pass api_key parameter when calling get_client()
            
            For more information, see README.md
            """.strip()
            raise ValueError(error_msg)

    return OpenAI(api_key=api_key, base_url=base_url)

