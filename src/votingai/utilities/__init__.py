"""
Utilities Module

Common utilities, configuration management, and shared type definitions
used throughout the voting system.
"""

# Configuration management
# Common types and constants
from .common_types import ConfigurationError, ErrorCodes, ProcessingError, SecurityError, VotingSystemError
from .configuration_management import DEFAULT_MODEL, LoggingConfiguration, ModelConfiguration, VotingSystemConfig
from .model_providers import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    ModelProvider,
    create_model_client,
    get_default_model,
)

__all__ = [
    # Configuration
    "VotingSystemConfig",
    "ModelConfiguration",
    "LoggingConfiguration",
    "DEFAULT_MODEL",
    # Model providers
    "ModelProvider",
    "create_model_client",
    "get_default_model",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_ANTHROPIC_MODEL",
    # Common types
    "VotingSystemError",
    "ConfigurationError",
    "SecurityError",
    "ProcessingError",
    "ErrorCodes",
]
