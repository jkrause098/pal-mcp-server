"""Tests for Gemini thinking_budget vs thinking_level request shaping."""

from unittest.mock import Mock, patch

from providers.gemini import GeminiModelProvider


def _enum_value(value):
    raw = getattr(value, "value", value)
    return raw.lower() if isinstance(raw, str) else raw


def _mock_gemini_client(mock_client_class):
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "ok"
    mock_candidate = Mock()
    mock_candidate.finish_reason = "STOP"
    mock_response.candidates = [mock_candidate]
    mock_usage = Mock()
    mock_usage.prompt_token_count = 1
    mock_usage.candidates_token_count = 1
    mock_response.usage_metadata = mock_usage
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client
    return mock_client


class TestGeminiThinkingConfig:
    @patch("google.genai.Client")
    def test_gemini4_uses_thinking_level(self, mock_client_class):
        mock_client = _mock_gemini_client(mock_client_class)
        provider = GeminiModelProvider(api_key="test-key")

        provider.generate_content(prompt="ping", model_name="gemini-4-argon", thinking_mode="medium")

        config = mock_client.models.generate_content.call_args.kwargs["config"]
        assert config.thinking_config is not None
        assert _enum_value(config.thinking_config.thinking_level) == "medium"
        assert config.thinking_config.thinking_budget is None

    @patch("google.genai.Client")
    def test_gemini38_flash_uses_thinking_level(self, mock_client_class):
        mock_client = _mock_gemini_client(mock_client_class)
        provider = GeminiModelProvider(api_key="test-key")

        provider.generate_content(prompt="ping", model_name="flash", thinking_mode="low")

        config = mock_client.models.generate_content.call_args.kwargs["config"]
        assert mock_client.models.generate_content.call_args.kwargs["model"] == "gemini-3.8-flash"
        assert _enum_value(config.thinking_config.thinking_level) == "low"
        assert config.thinking_config.thinking_budget is None

    @patch("google.genai.Client")
    def test_gemini25_uses_thinking_budget(self, mock_client_class):
        mock_client = _mock_gemini_client(mock_client_class)
        provider = GeminiModelProvider(api_key="test-key")

        provider.generate_content(prompt="ping", model_name="gemini-2.5-pro", thinking_mode="medium")

        config = mock_client.models.generate_content.call_args.kwargs["config"]
        assert config.thinking_config.thinking_budget == int(32768 * 0.33)
        assert _enum_value(config.thinking_config.thinking_level) is None
