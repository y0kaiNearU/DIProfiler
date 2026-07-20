from __future__ import annotations

from typing import Any, Protocol

from core.profiler import Profiler
from models.models import (
    DatabaseSource,
    EngineRecommendation,
    EngineType,
    FileSource,
    PipelineRequest,
    ProfilingResult,
)


class LLMEngineClient(Protocol):
    """Anything that can turn a formatted pipeline description into ranked engine recommendations."""

    def recommend(self, prompt: str, engine_options: list[str]) -> list[dict[str, Any]]:
        """Return a list of {"engine": str, "confidence": float, "reasoning": str} dicts,
        each "engine" one of engine_options."""
        ...


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


class LLMEngineProfiler(Profiler[EngineRecommendation]):
    """
    Engine profiler backed by an LLM client of your choice.

    Formats the PipelineRequest as a structured prompt, asks the client to rank
    the available engines, and returns the response as a ProfilingResult.
    Defaults to Claude (Anthropic API) if no client is given; swap in any
    LLMEngineClient implementation — a different provider, a stub for tests —
    via the constructor.

    Args:
        client: An LLMEngineClient. Defaults to AnthropicLLMClient() if omitted
                (requires `uv add anthropic` and ANTHROPIC_API_KEY set).
    """

    def __init__(self, client: LLMEngineClient | None = None) -> None:
        self._client = client

    def _get_client(self) -> LLMEngineClient:
        if self._client is None:
            from profilers.engine.anthropic_llm_client import AnthropicLLMClient
            self._client = AnthropicLLMClient()
        return self._client

    @property
    def name(self) -> str:
        return "llm_engine_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return True

    def profile(self, request: PipelineRequest) -> ProfilingResult[EngineRecommendation]:
        client = self._get_client()
        prompt = _format_request(request)
        available = set(request.available_engines)
        engine_options = [e.value for e in request.available_engines]

        raw = client.recommend(prompt, engine_options)

        recommendations = [
            EngineRecommendation(
                engine=EngineType(item["engine"]),
                confidence=round(float(item["confidence"]), 3),
                reasoning=item["reasoning"],
            )
            for item in raw
            if EngineType(item["engine"]) in available
        ]
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return ProfilingResult(request=request, recommendations=recommendations)
