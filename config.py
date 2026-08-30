"""
config.py — Application-wide configuration loaded from environment variables.

This module uses pydantic-settings to define a strongly-typed Settings class
whose fields are automatically populated from the process environment or a
".env" file in the project root. Centralising configuration here means that
every other module simply imports the pre-built `settings` singleton rather
than calling os.environ directly, making the configuration surface easy to
audit and extend. Any required field that is missing at start-up will raise a
clear validation error before the app accepts any traffic.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
# BaseSettings extends Pydantic's BaseModel with the ability to read field
# values from environment variables.
# SettingsConfigDict provides the configuration knobs for BaseSettings itself.


class Settings(BaseSettings):
    """Typed container for all runtime configuration values.

    Fields that have no default are *required* — the app will refuse to start
    if they are absent from the environment or .env file.
    """

    # Tell pydantic-settings where to look for a .env file.
    # extra="ignore" means unrecognised env vars are silently skipped rather
    # than raising a validation error, which is friendly for shared environments.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- YouTube Data API ---
    youtube_api_key: str  # Required. Obtained from Google Cloud Console (Data API v3)

    # --- LLM routing ---
    llm_provider: str  # Which provider to use: "anthropic", "gemini", "groq", or "huggingface"
    llm_model: str     # The exact model name/ID expected by the chosen provider's SDK

    # --- Provider-specific credentials (all optional; only the active one is needed) ---
    anthropic_api_key: str | None = None       # Required when llm_provider == "anthropic"
    google_api_key: str | None = None          # Required when llm_provider == "gemini"
    groq_api_key: str | None = None            # Required when llm_provider == "groq"
    huggingfacehub_api_token: str | None = None  # Required when llm_provider == "huggingface"

    # --- Phase 0: request-level safety ceiling ---
    # Overall wall-clock budget (seconds) for a single /classify-video request,
    # covering both YouTube pagination and LLM classification combined. This is
    # a safety ceiling against a pathological video (extremely high comment
    # count) running unbounded — it is not a comment-count cap: pagination still
    # fully exhausts within this time budget, per CLAUDE.md's pagination rule.
    request_timeout_ceiling_seconds: int = 900  # 15 minutes

    # --- Phase 1: response caching ---
    # How long a classify-video response is cached in-memory, keyed by video_id.
    cache_ttl_seconds: int = 3600  # 1 hour


# Module-level singleton: import `settings` directly anywhere in the project.
# Instantiating here triggers immediate validation so misconfiguration is caught
# at import time rather than when the first request arrives.
settings = Settings()
