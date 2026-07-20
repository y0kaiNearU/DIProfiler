import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from profilers.engine.llm_clients.openai_client import OpenAILLMClient


def _mock_client(recommendations: list[dict]):
    """Build a fake openai client that returns the given recommendations."""
    tool_call = MagicMock()
    tool_call.function.arguments = json.dumps({"recommendations": recommendations})

    message = MagicMock()
    message.tool_calls = [tool_call]

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_client_initialized_lazily():
    client = OpenAILLMClient()
    assert client._client is None


def test_recommend_returns_raw_recommendations():
    client = OpenAILLMClient()
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
    client = OpenAILLMClient(model="gpt-4o-mini")
    client._client = _mock_client([{"engine": "duckdb", "confidence": 1.0, "reasoning": "r"}])

    client.recommend("prompt text", ["duckdb"])

    call_kwargs = client._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["messages"][-1] == {"role": "user", "content": "prompt text"}


def test_recommend_scopes_tool_schema_to_engine_options():
    client = OpenAILLMClient()
    client._client = _mock_client([{"engine": "duckdb", "confidence": 1.0, "reasoning": "r"}])

    client.recommend("prompt text", ["duckdb", "polars"])

    call_kwargs = client._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"]["function"]["name"] == "recommend_engines"
    tool = next(t for t in call_kwargs["tools"] if t["function"]["name"] == "recommend_engines")
    engine_enum = tool["function"]["parameters"]["properties"]["recommendations"]["items"]["properties"]["engine"]["enum"]
    assert engine_enum == ["duckdb", "polars"]


def test_import_error_when_openai_missing():
    client = OpenAILLMClient()

    with patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(ImportError, match="openai"):
            client._get_client()
