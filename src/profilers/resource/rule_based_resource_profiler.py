from __future__ import annotations

import math

from core.profiler import Profiler
from models.models import PipelineRequest, ProfilingResult, ResourceRecommendation

_GB = 1024 ** 3
_MEMORY_HEADROOM_FACTOR = 3.0  # engines typically need several times the raw data size in working memory
_DEFAULT_CORES_FRACTION = 0.25  # fraction of available cores to use when dataset size is unknown
_DEFAULT_MEMORY_FRACTION = 0.25  # fraction of available memory to use when dataset size is unknown


def _recommend_cores(size_bytes: int | None, available_cores: int) -> tuple[int, str]:
    if size_bytes is None:
        cores = max(1, round(available_cores * _DEFAULT_CORES_FRACTION))
        return cores, f"no size information; defaulting to {cores} of {available_cores} available cores"
    size_gb = size_bytes / _GB
    cores = min(available_cores, max(1, math.ceil(size_gb)))
    return cores, f"dataset size {size_gb:.2f} GB scaled against {available_cores} available cores"


def _recommend_memory(size_bytes: int | None, available_memory_bytes: int) -> tuple[int, str]:
    if size_bytes is None:
        memory = max(1, round(available_memory_bytes * _DEFAULT_MEMORY_FRACTION))
        return memory, (
            f"no size information; defaulting to {memory / _GB:.2f} GB of "
            f"{available_memory_bytes / _GB:.2f} GB available"
        )
    needed = int(size_bytes * _MEMORY_HEADROOM_FACTOR)
    memory = min(available_memory_bytes, needed)
    return memory, (
        f"dataset size {size_bytes / _GB:.2f} GB needs roughly {needed / _GB:.2f} GB of working memory "
        f"({_MEMORY_HEADROOM_FACTOR:.0f}x headroom), capped at {available_memory_bytes / _GB:.2f} GB available"
    )


class RuleBasedResourceProfiler(Profiler[ResourceRecommendation]):
    """
    Recommends a core count and memory allocation for a request, sized against
    PipelineRequest.available_cores/available_memory_bytes and the dataset's
    size_bytes. Falls back to a conservative fraction of the available budget
    when size_bytes is unknown (reflected in a lower confidence).

    Translating cores/memory_bytes into an engine's own knobs (e.g. Spark's
    executor.instances/executor.cores, Dask's n_workers/threads_per_worker) is
    left to the caller — this profiler only sizes the total budget to use.
    """

    @property
    def name(self) -> str:
        return "rule_based_resource_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return request.available_cores is not None and request.available_memory_bytes is not None

    def profile(self, request: PipelineRequest) -> ProfilingResult[ResourceRecommendation]:
        size_bytes = request.source.size_bytes

        cores, cores_reason = _recommend_cores(size_bytes, request.available_cores)
        memory, memory_reason = _recommend_memory(size_bytes, request.available_memory_bytes)
        confidence = 1.0 if size_bytes is not None else 0.4

        recommendation = ResourceRecommendation(
            cores=cores,
            memory_bytes=memory,
            confidence=confidence,
            reasoning=f"{cores_reason}; {memory_reason}",
        )

        return ProfilingResult(request=request, recommendations=[recommendation])
