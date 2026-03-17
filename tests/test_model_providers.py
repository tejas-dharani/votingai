"""Tests for model provider utilities."""

import pytest

from votingai.utilities.model_providers import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    ModelProvider,
    create_model_client,
    get_default_model,
)
from votingai.utilities.configuration_management import ModelConfiguration


# ---------------------------------------------------------------------------
# ModelProvider enum
# ---------------------------------------------------------------------------

class TestModelProvider:
    def test_openai_value(self):
        assert ModelProvider.OPENAI == "openai"

    def test_anthropic_value(self):
        assert ModelProvider.ANTHROPIC == "anthropic"

    def test_all_providers_present(self):
        values = [p.value for p in ModelProvider]
        assert "openai" in values
        assert "anthropic" in values

    def test_provider_from_string(self):
        assert ModelProvider("openai") == ModelProvider.OPENAI
        assert ModelProvider("anthropic") == ModelProvider.ANTHROPIC

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            ModelProvider("gemini")


# ---------------------------------------------------------------------------
# Default model constants
# ---------------------------------------------------------------------------

class TestDefaultModels:
    def test_openai_default_is_set(self):
        assert isinstance(DEFAULT_OPENAI_MODEL, str)
        assert len(DEFAULT_OPENAI_MODEL) > 0

    def test_anthropic_default_is_set(self):
        assert isinstance(DEFAULT_ANTHROPIC_MODEL, str)
        assert len(DEFAULT_ANTHROPIC_MODEL) > 0

    def test_defaults_are_different(self):
        assert DEFAULT_OPENAI_MODEL != DEFAULT_ANTHROPIC_MODEL

    def test_anthropic_default_contains_claude(self):
        assert "claude" in DEFAULT_ANTHROPIC_MODEL.lower()


# ---------------------------------------------------------------------------
# get_default_model
# ---------------------------------------------------------------------------

class TestGetDefaultModel:
    def test_openai_returns_openai_model(self):
        model = get_default_model(ModelProvider.OPENAI)
        assert model == DEFAULT_OPENAI_MODEL

    def test_anthropic_returns_anthropic_model(self):
        model = get_default_model(ModelProvider.ANTHROPIC)
        assert model == DEFAULT_ANTHROPIC_MODEL

    def test_invalid_provider_raises(self):
        with pytest.raises((ValueError, KeyError)):
            get_default_model("invalid")  # type: ignore


# ---------------------------------------------------------------------------
# create_model_client — error handling (no API keys needed)
# ---------------------------------------------------------------------------

class TestCreateModelClientErrors:
    def test_invalid_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_model_client("invalid")  # type: ignore


# ---------------------------------------------------------------------------
# ModelConfiguration
# ---------------------------------------------------------------------------

class TestModelConfiguration:
    def test_default_provider_is_openai(self):
        config = ModelConfiguration()
        assert config.provider == ModelProvider.OPENAI

    def test_for_openai_factory(self):
        config = ModelConfiguration.for_openai()
        assert config.provider == ModelProvider.OPENAI
        assert "gpt" in config.default_model.lower()

    def test_for_anthropic_factory(self):
        config = ModelConfiguration.for_anthropic()
        assert config.provider == ModelProvider.ANTHROPIC
        assert "claude" in config.default_model.lower()

    def test_for_openai_custom_model(self):
        config = ModelConfiguration.for_openai(model="gpt-4o")
        assert config.default_model == "gpt-4o"

    def test_for_anthropic_custom_model(self):
        config = ModelConfiguration.for_anthropic(model="claude-opus-4-6")
        assert config.default_model == "claude-opus-4-6"

    def test_from_environment_defaults_to_openai(self, monkeypatch):
        monkeypatch.delenv("VOTINGAI_PROVIDER", raising=False)
        config = ModelConfiguration.from_environment()
        assert config.provider == ModelProvider.OPENAI

    def test_from_environment_anthropic(self, monkeypatch):
        monkeypatch.setenv("VOTINGAI_PROVIDER", "anthropic")
        config = ModelConfiguration.from_environment()
        assert config.provider == ModelProvider.ANTHROPIC

    def test_from_environment_invalid_provider_falls_back_to_openai(self, monkeypatch):
        monkeypatch.setenv("VOTINGAI_PROVIDER", "gemini")
        config = ModelConfiguration.from_environment()
        assert config.provider == ModelProvider.OPENAI

    def test_from_environment_custom_model(self, monkeypatch):
        monkeypatch.setenv("VOTINGAI_DEFAULT_MODEL", "gpt-4o")
        monkeypatch.delenv("VOTINGAI_PROVIDER", raising=False)
        config = ModelConfiguration.from_environment()
        assert config.default_model == "gpt-4o"
