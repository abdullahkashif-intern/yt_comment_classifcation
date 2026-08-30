"""
llm/provider.py — LLM provider factory: returns a LangChain chat model instance.

This module is the single place where LLM provider selection and SDK-specific
configuration live. Its one public function, get_chat_model(), reads the
llm_provider and llm_model settings from the global `settings` singleton and
returns a fully-initialised LangChain BaseChatModel that the rest of the app
(specifically llm/classifier.py) can call uniformly via the LangChain interface.

Keeping provider wiring isolated here means adding a new provider requires
touching only this file, and all SDK imports are lazy (inside if-blocks) so
unused provider packages are never imported at runtime.

Supported providers: anthropic, gemini, groq, huggingface.
"""

from langchain_core.language_models.chat_models import BaseChatModel
# BaseChatModel is the abstract LangChain base class for all chat-capable models.
# Returning this type (rather than a concrete class) keeps the rest of the code
# provider-agnostic — any model that implements BaseChatModel works identically.

from config import settings  # Typed singleton holding all env-var configuration

# Hard timeout applied to every provider request.
# A misconfigured API key or a network stall should surface quickly as an error
# rather than hanging the Uvicorn worker thread indefinitely.
# Each provider SDK exposes this under a different parameter name (see below).
REQUEST_TIMEOUT_SECONDS = 60


def get_chat_model() -> BaseChatModel:
    """Instantiate and return a LangChain chat model for the configured LLM provider.

    Reads `settings.llm_provider` (case-insensitive) to select which SDK to
    initialise, and `settings.llm_model` for the exact model name/ID.

    Returns:
        A fully-configured BaseChatModel ready to call `.invoke()` or
        `.with_structured_output()` on.

    Raises:
        ValueError: If `settings.llm_provider` is not a recognised provider name.
    """
    # Normalise to lowercase so "Anthropic" and "ANTHROPIC" both match
    provider = settings.llm_provider.lower()

    # ------------------------------------------------------------------
    # Anthropic (Claude family)
    # SDK parameter for timeout is `default_request_timeout`.
    # ------------------------------------------------------------------
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # Lazy import — only loaded when needed

        return ChatAnthropic(
            model=settings.llm_model,                       # e.g. "claude-3-5-sonnet-20241022"
            anthropic_api_key=settings.anthropic_api_key,   # From ANTHROPIC_API_KEY env var
            default_request_timeout=REQUEST_TIMEOUT_SECONDS,
        )

    # ------------------------------------------------------------------
    # Google Gemini
    # SDK parameter for timeout is `timeout`.
    # ------------------------------------------------------------------
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # Lazy import

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,               # e.g. "gemini-1.5-pro"
            google_api_key=settings.google_api_key, # From GOOGLE_API_KEY env var
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    # ------------------------------------------------------------------
    # Groq (hosted open-source models via Groq Cloud)
    # SDK parameter for timeout is `request_timeout`.
    # ------------------------------------------------------------------
    if provider == "groq":
        from langchain_groq import ChatGroq  # Lazy import

        # Special-case: Groq's "openai/gpt-oss" models default to
        # reasoning_effort="medium", which can spend so many output tokens on
        # chain-of-thought that the structured-output JSON gets truncated for
        # a full 25-comment batch. Switch to "low" effort and cap max_tokens.
        reasoning_kwargs = {}
        if settings.llm_model.startswith("openai/gpt-oss"):
            reasoning_kwargs = {"reasoning_effort": "low", "max_tokens": 4096}

        return ChatGroq(
            model_name=settings.llm_model,          # e.g. "llama3-70b-8192"
            groq_api_key=settings.groq_api_key,     # From GROQ_API_KEY env var
            request_timeout=REQUEST_TIMEOUT_SECONDS,
            **reasoning_kwargs,                     # Unpacked only for gpt-oss models
        )

    # ------------------------------------------------------------------
    # Hugging Face Inference Endpoints
    # The LangChain wrapper requires a two-step setup: first create an
    # HuggingFaceEndpoint (handles HTTP calls), then wrap it in
    # ChatHuggingFace (adds the chat message formatting layer).
    # ------------------------------------------------------------------
    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # Lazy import

        # HuggingFaceEndpoint points at a specific model repository on the Hub
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.llm_model,                           # e.g. "mistralai/Mistral-7B-Instruct-v0.2"
            huggingfacehub_api_token=settings.huggingfacehub_api_token,  # From HUGGINGFACEHUB_API_TOKEN env var
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        # Wrap the raw endpoint in a chat-message-aware adapter
        return ChatHuggingFace(llm=endpoint)

    # ------------------------------------------------------------------
    # Unrecognised provider — surface a clear actionable error message
    # ------------------------------------------------------------------
    raise ValueError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Must be one of: anthropic, gemini, groq, huggingface."
    )
