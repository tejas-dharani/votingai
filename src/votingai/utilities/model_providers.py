"""
Model Providers

Factory utilities for creating model clients for both OpenAI and Anthropic (Claude).
Supports drop-in provider switching for VotingAI agent teams.
"""

import os
from enum import Enum
from typing import Any, Optional


class ModelProvider(str, Enum):
    """Supported AI model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Default models per provider
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def create_model_client(
    provider: ModelProvider,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a model client for the specified provider.

    The returned client is compatible with autogen's AssistantAgent and
    can be passed directly to any VotingAI agent.

    Args:
        provider: ModelProvider.OPENAI or ModelProvider.ANTHROPIC
        model: Model name. Defaults to provider's recommended model.
        api_key: API key. If None, reads from environment variable:
                 - OpenAI:    OPENAI_API_KEY
                 - Anthropic: ANTHROPIC_API_KEY
        **kwargs: Additional keyword arguments passed to the underlying client.

    Returns:
        A model client compatible with autogen's ChatAgent.

    Examples:
        # OpenAI
        client = create_model_client(ModelProvider.OPENAI, model="gpt-4o")

        # Anthropic (Claude)
        client = create_model_client(ModelProvider.ANTHROPIC, model="claude-opus-4-6")

        # Use environment variable for API key automatically
        client = create_model_client(ModelProvider.ANTHROPIC)
    """
    if provider == ModelProvider.OPENAI:
        try:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except ImportError as e:
            raise ImportError(
                "OpenAI support requires: pip install 'autogen-ext[openai]'"
            ) from e

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        init_kwargs: dict[str, Any] = {"model": model or DEFAULT_OPENAI_MODEL, **kwargs}
        if resolved_key:
            init_kwargs["api_key"] = resolved_key
        return OpenAIChatCompletionClient(**init_kwargs)

    elif provider == ModelProvider.ANTHROPIC:
        try:
            from autogen_ext.models.anthropic import AnthropicChatCompletionClient
        except ImportError as e:
            raise ImportError(
                "Anthropic support requires: pip install 'autogen-ext[anthropic]'"
            ) from e

        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        init_kwargs = {"model": model or DEFAULT_ANTHROPIC_MODEL, **kwargs}
        if resolved_key:
            init_kwargs["api_key"] = resolved_key
        return AnthropicChatCompletionClient(**init_kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}. Choose ModelProvider.OPENAI or ModelProvider.ANTHROPIC.")


def get_default_model(provider: ModelProvider) -> str:
    """Return the default model name for a given provider."""
    if provider == ModelProvider.OPENAI:
        return DEFAULT_OPENAI_MODEL
    elif provider == ModelProvider.ANTHROPIC:
        return DEFAULT_ANTHROPIC_MODEL
    raise ValueError(f"Unknown provider: {provider}")
