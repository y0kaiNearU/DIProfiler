from __future__ import annotations

import json
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
        "type": "function",
        "function": {
            "name": _RECOMMEND_TOOL_NAME,
            "description": (
                "Return a ranked list of engine recommendations for the described data pipeline. "
                "Only include engines from the available options. "
                "Confidence values must sum to 1.0."
            ),
            "parameters": {
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
        },
    }


class OpenAILLMClient:
    """
    LLMEngineClient implementation backed by OpenAI (Chat Completions API).

    Requires: uv add openai  (or pip install openai)
    The OPENAI_API_KEY environment variable must be set.

    Args:
        model:   OpenAI model ID. Defaults to gpt-4o.
        api_key: Overrides OPENAI_API_KEY env var if provided.
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "OpenAILLMClient requires the 'openai' package. "
                    "Install it with: uv add openai"
                ) from e
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def recommend(self, prompt: str, engine_options: list[str]) -> list[dict[str, Any]]:
        client = self._get_client()

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[_build_tool(engine_options)],
            tool_choice={"type": "function", "function": {"name": _RECOMMEND_TOOL_NAME}},
        )

        tool_call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        return arguments["recommendations"]
