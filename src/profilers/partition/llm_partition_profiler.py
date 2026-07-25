from __future__ import annotations

from typing import Any, Protocol

from core.profiler import Profiler
from models.models import (
    DatabaseSource,
    FileSource,
    PartitionRecommendation,
    PipelineRequest,
    ProfilingResult,
)


class LLMPartitionClient(Protocol):
    """Anything that can turn a formatted pipeline description into ranked partition-column recommendations."""

    def recommend(self, prompt: str, column_options: list[str]) -> list[dict[str, Any]]:
        """Return a list of {"column": str, "confidence": float, "reasoning": str} dicts,
        each "column" one of column_options."""
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

    lines.append(f"  Schema: {src_info.schema}")

    if request.operations:
        lines.append(f"  Operations: {[op.value for op in request.operations]}")

    if request.destination:
        dst = request.destination.source
        if isinstance(dst, FileSource):
            lines.append(f"  Destination: {dst.format.value} file at '{dst.path}'")
        elif isinstance(dst, DatabaseSource):
            lines.append(f"  Destination: {dst.database_type} table '{dst.table_name}'")

    lines.append(f"  Candidate partition columns: {list(src_info.schema)}")
    return "\n".join(lines)


class LLMPartitionProfiler(Profiler[PartitionRecommendation]):
    """
    Partition profiler backed by an LLM client of your choice.

    Formats the PipelineRequest's schema and workload as a structured prompt,
    asks the client to rank candidate partition columns, and returns the
    response as a ProfilingResult. Defaults to Claude (Anthropic API) if no
    client is given; swap in any LLMPartitionClient implementation — a
    different provider, a stub for tests — via the constructor.

    Args:
        client: An LLMPartitionClient. Defaults to AnthropicLLMClient() if omitted
                (requires `uv add anthropic` and ANTHROPIC_API_KEY set).
    """

    def __init__(self, client: LLMPartitionClient | None = None) -> None:
        self._client = client

    def _get_client(self) -> LLMPartitionClient:
        if self._client is None:
            from profilers.partition.llm_clients.anthropic_client import AnthropicLLMClient
            self._client = AnthropicLLMClient()
        return self._client

    @property
    def name(self) -> str:
        return "llm_partition_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return bool(request.source.schema)

    def profile(self, request: PipelineRequest) -> ProfilingResult[PartitionRecommendation]:
        client = self._get_client()
        prompt = _format_request(request)
        schema = request.source.schema
        column_options = list(schema)

        raw = client.recommend(prompt, column_options)

        recommendations = [
            PartitionRecommendation(
                column=item["column"],
                confidence=round(float(item["confidence"]), 3),
                reasoning=item["reasoning"],
            )
            for item in raw
            if item["column"] in schema
        ]
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return ProfilingResult(request=request, recommendations=recommendations)
