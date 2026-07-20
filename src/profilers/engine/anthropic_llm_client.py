from __future__ import annotations

from typing import Any

_RECOMMEND_TOOL_NAME = "recommend_engines"

_SYSTEM_PROMPT = """\
You are an expert data engineering advisor. Given a description of a data pipeline request,
recommend which processing engine(s) are best suited, from the available options only.
Consider: dataset size, row count, file format ecosystem (Arrow, Hadoop, etc.), operation
complexity (joins, windows), and typical single-node vs. distributed tradeoffs.
"""


def _build_tool(engine_options: list[str]) -> dict[str, Any]:
    return {
        "name": _RECOMMEND_TOOL_NAME,
        "description": (
            "Return a ranked list of engine recommendations for the described data pipeline. "
            "Only include engines from the available options. "
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
                            "engine": {
                                "type": "string",
                                "enum": engine_options,
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
                        "required": ["engine", "confidence", "reasoning"],
                    },
                }
            },
            "required": ["recommendations"],
        },
    }


class AnthropicLLMClient:
    """
    LLMEngineClient implementation backed by Claude (Anthropic API).

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

    def recommend(self, prompt: str, engine_options: list[str]) -> list[dict[str, Any]]:
        client = self._get_client()

        response = client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=[_build_tool(engine_options)],
            tool_choice={"type": "tool", "name": _RECOMMEND_TOOL_NAME},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")
        return tool_block.input["recommendations"]
