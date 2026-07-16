from __future__ import annotations

from typing import Any

from core.profiler import Profiler
from models.models import (
    DatabaseSource,
    EngineRecommendation,
    EngineType,
    FileSource,
    PipelineRequest,
    ProfilingResult,
)

_RECOMMEND_TOOL = {
    "name": "recommend_engines",
    "description": (
        "Return a ranked list of engine recommendations for the described data pipeline. "
        "Only include engines from the available_engines list. "
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
                            "enum": [e.value for e in EngineType],
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0.0–1.0; all confidences must sum to 1.0",
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

_SYSTEM_PROMPT = """\
You are an expert data engineering advisor. Given a description of a data pipeline request,
recommend which processing engine(s) are best suited, from the available options only.
Consider: dataset size, row count, file format ecosystem (Arrow, Hadoop, etc.), operation
complexity (joins, windows), and typical single-node vs. distributed tradeoffs.
"""


def _format_request(request: PipelineRequest) -> str:
    src_info = request.source
    src = src_info.source

    lines: list[str] = ["Data pipeline request:"]

    if isinstance(src, FileSource):
        lines.append(f"  Source: {src.format.value} file at '{src.path}'")
    elif isinstance(src, DatabaseSource):
        lines.append(f"  Source: {src.database_type} database, table '{src.table_name}'")

    if src_info.size_bytes is not None:
        gb = src_info.size_bytes / (1024 ** 3)
        lines.append(f"  Size: {gb:.3f} GB ({src_info.size_bytes:,} bytes)")
    if src_info.row_count is not None:
        lines.append(f"  Rows: {src_info.row_count:,}")
    if src_info.num_columns is not None:
        lines.append(f"  Columns: {src_info.num_columns}")
    if src_info.schema:
        lines.append(f"  Schema sample: {dict(list(src_info.schema.items())[:8])}")

    if request.operations:
        lines.append(f"  Operations: {[op.value for op in request.operations]}")

    if request.destination:
        dst = request.destination.source
        if isinstance(dst, FileSource):
            lines.append(f"  Destination: {dst.format.value} file at '{dst.path}'")
        elif isinstance(dst, DatabaseSource):
            lines.append(f"  Destination: {dst.database_type} table '{dst.table_name}'")

    lines.append(f"  Available engines: {[e.value for e in request.available_engines]}")
    return "\n".join(lines)


class LLMEngineProfiler(Profiler):
    """
    Engine profiler backed by Claude (Anthropic API).

    Formats the PipelineRequest as a structured prompt, asks Claude to rank the
    available engines, and returns the response as a ProfilingResult.

    Requires: uv add anthropic  (or pip install anthropic)
    The ANTHROPIC_API_KEY environment variable must be set.

    Args:
        model:   Claude model ID. Defaults to claude-opus-4-8.
        api_key: Overrides ANTHROPIC_API_KEY env var if provided.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "LLMEngineProfiler requires the 'anthropic' package. "
                    "Install it with: uv add anthropic"
                ) from e
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "llm_engine_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return True

    def profile(self, request: PipelineRequest) -> ProfilingResult:
        client = self._get_client()

        response = client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=[_RECOMMEND_TOOL],
            tool_choice={"type": "tool", "name": "recommend_engines"},
            messages=[{"role": "user", "content": _format_request(request)}],
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")
        available = set(request.available_engines)

        recommendations = [
            EngineRecommendation(
                engine=EngineType(item["engine"]),
                confidence=round(float(item["confidence"]), 3),
                reasoning=item["reasoning"],
            )
            for item in tool_block.input["recommendations"]
            if EngineType(item["engine"]) in available
        ]
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return ProfilingResult(request=request, recommendations=recommendations)
