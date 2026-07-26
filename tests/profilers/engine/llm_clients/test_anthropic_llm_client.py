import sys
from unittest.mock import MagicMock, patch

import pytest

from profilers.engine.llm_clients.anthropic_client import AnthropicLLMClient


def _mock_client(recommendations: list[dict]):
    """Build a fake anthropic client that returns the given recommendations."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"recommendations": recommendations}

    response = MagicMock()
    response.content = [tool_block]

    client = MagicMock()
    client.messages.create.return_value = response
    return client


def test_client_initialized_lazily():
    client = AnthropicLLMClient()
    assert client._client is None


def test_recommend_returns_raw_recommendations():
    client = AnthropicLLMClient()
    client._client = _mock_client([
        {"engine": "duckdb", "confidence": 0.7, "reasoning": "small file"},
        {"engine": "spark", "confidence": 0.3, "reasoning": "r"},
    ])

    result = client.recommend("some prompt", ["duckdb", "spark"])

    assert result == [
        {"engine": "duckdb", "confidence": 0.7, "reasoning": "small file"},
        {"engine": "spark", "confidence": 0.3, "reasoning": "r"},
    ]


def test_recommend_passes_model_and_prompt():
    client = AnthropicLLMClient(model="claude-opus-4-8")
    client._client = _mock_client([{"engine": "duckdb", "confidence": 1.0, "reasoning": "r"}])

    client.recommend("prompt text", ["duckdb"])

    call_kwargs = client._client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-8"
    assert call_kwargs["messages"] == [{"role": "user", "content": "prompt text"}]


def test_recommend_scopes_tool_schema_to_engine_options():
    client = AnthropicLLMClient()
    client._client = _mock_client([{"engine": "duckdb", "confidence": 1.0, "reasoning": "r"}])

    client.recommend("prompt text", ["duckdb", "polars"])

    call_kwargs = client._client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"]["name"] == "recommend_engines"
    tool = next(t for t in call_kwargs["tools"] if t["name"] == "recommend_engines")
    engine_enum = tool["input_schema"]["properties"]["recommendations"]["items"]["properties"]["engine"]["enum"]
    assert engine_enum == ["duckdb", "polars"]


def test_import_error_when_anthropic_missing():
    client = AnthropicLLMClient()

    with patch.dict(sys.modules, {"anthropic": None}):
        with pytest.raises(ImportError, match="anthropic"):
            client._get_client()
