from __future__ import annotations

import dataclasses
from typing import Callable

from core.profiler import Profiler
from models.models import DatasetInfo, PipelineRequest, ProfilingResult


class PrefetchingProfiler(Profiler):
    """
    Profiler wrapper that enriches a PipelineRequest with real dataset metadata
    before delegating to an inner profiler.

    Use when PipelineRequest.source lacks size_bytes / row_count — the prefetch_fn
    is responsible for fetching those stats (read from disk, query a catalog, call
    an API, etc.) and returning an updated DatasetInfo.

    Args:
        prefetch_fn: Callable that takes a PipelineRequest and returns a DatasetInfo
                     with size_bytes, row_count, and num_columns filled in.
        delegate:    Profiler to delegate to after enrichment. Defaults to
                     RuleBasedEngineProfiler if not provided.
    """

    def __init__(
        self,
        prefetch_fn: Callable[[PipelineRequest], DatasetInfo],
        delegate: Profiler | None = None,
    ) -> None:
        self._prefetch_fn = prefetch_fn
        self._delegate = delegate

    def _get_delegate(self) -> Profiler:
        if self._delegate is None:
            from profilers.engine_profiler import RuleBasedEngineProfiler
            self._delegate = RuleBasedEngineProfiler()
        return self._delegate

    @property
    def name(self) -> str:
        return "prefetching_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        src = request.source
        return src.size_bytes is None or src.row_count is None

    def profile(self, request: PipelineRequest) -> ProfilingResult:
        enriched_info = self._prefetch_fn(request)
        enriched_request = dataclasses.replace(request, source=enriched_info)
        return self._get_delegate().profile(enriched_request)
