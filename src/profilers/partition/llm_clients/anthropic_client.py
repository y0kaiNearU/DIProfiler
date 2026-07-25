from __future__ import annotations

from typing import Any

_RECOMMEND_TOOL_NAME = "recommend_partition_columns"

_SYSTEM_PROMPT = """\
You are an expert data engineering advisor. Given a description of a data pipeline request,
including its column schema, recommend which column(s) are best suited as a partition key,
from the candidate columns only. Favor temporal columns (dates/timestamps) and low-cardinality
categorical columns; avoid identifier-like columns (ids, uuids/guids) and high-cardinality
columns, which cause excessive small partitions.
"""


def _build_tool(column_options: list[str]) -> dict[str, Any]:
    return {
        "name": _RECOMMEND_TOOL_NAME,
        "description": (
            "Return a ranked list of partition-column recommendations for the described data pipeline. "
            "Only include columns from the candidate options. "
            "Confidence values must sum to 1.0."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string",
                                "enum": column_options,
                            },
                            "confidence": {
                                "type": "number",
                                "description": "0.0-1.0; all confidences must sum to 1.0",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "One-sentence justification.",
                            },
                        },
                        "required": ["column", "confidence", "reasoning"],
                    },
                }
            },
            "required": ["recommendations"],
        },
    }


class AnthropicLLMClient:
    """
    LLMPartitionClient implementation backed by Claude (Anthropic API).

    Requires: uv add anthropic  (or pip install anthropic)
    The ANTHROPIC_API_KEY environment variable must be set.

    Args:
        model:   Claude model ID. Defaults to claude-opus-4-8.
        api_key: Overrides ANTHROPIC_API_KEY env var if provided.
    """

    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "AnthropicLLMClient requires the 'anthropic' package. "
                    "Install it with: uv add anthropic"
                ) from e
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def recommend(self, prompt: str, column_options: list[str]) -> list[dict[str, Any]]:
        client = self._get_client()

        response = client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=[_build_tool(column_options)],
            tool_choice={"type": "tool", "name": _RECOMMEND_TOOL_NAME},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")
        recommendations: list[dict[str, Any]] = tool_block.input["recommendations"]
        return recommendations
