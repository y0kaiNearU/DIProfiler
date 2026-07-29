from __future__ import annotations

from typing import Callable, Hashable, Literal

from core.profiler import Profiler
from models.models import PipelineRequest, ProfilingResult, Recommendation
from profilers.common.voting import Tally

EnsembleMode = Literal["weighted_sum", "max"]


class EnsembleProfiler[R: Recommendation, K: Hashable](Profiler[R]):
    """
    Combines multiple profilers into one recommendation set.

    Each sub-profiler that can_handle the request contributes its
    recommendations, optionally scaled by its weight and filtered by filter_fn.
    key_fn extracts the candidate identity recommendations are grouped by
    (e.g. `lambda rec: rec.engine`, `lambda rec: rec.column`).

    mode="weighted_sum" (default): confidences are summed per candidate across
        profilers and renormalized to sum to 1.0. Requires build_fn to
        construct the merged recommendation from (candidate, confidence, reasoning).
    mode="max": the single highest-(weight * confidence) recommendation per
        candidate wins, unchanged, rather than being combined with the rest.
        build_fn is not needed.

    Usage:
        profiler = EnsembleProfiler(
            [RuleBasedEngineProfiler(), MLEngineProfiler(model)],
            key_fn=lambda rec: rec.engine,
            build_fn=lambda engine, confidence, reasoning: EngineRecommendation(engine, confidence, reasoning),
        )
        profiler = EnsembleProfiler([a, b], key_fn=lambda rec: rec.engine, mode="max")
    """

    def __init__(
        self,
        profilers: list[Profiler[R]],
        key_fn: Callable[[R], K],
        weights: list[float] | None = None,
        mode: EnsembleMode = "weighted_sum",
        build_fn: Callable[[K, float, str], R] | None = None,
        filter_fn: Callable[[R], bool] | None = None,
    ) -> None:
        if weights is not None and len(weights) != len(profilers):
            raise ValueError("weights must have the same length as profilers")
        if mode not in ("weighted_sum", "max"):
            raise ValueError(f"unknown mode: {mode!r}")
        if mode == "weighted_sum" and build_fn is None:
            raise ValueError("mode='weighted_sum' requires build_fn to construct merged recommendations")
        self._profilers = profilers
        self._key_fn = key_fn
        self._weights = weights if weights is not None else [1.0] * len(profilers)
        self._mode = mode
        self._build_fn = build_fn
        self._filter_fn = filter_fn

    @property
    def name(self) -> str:
        return "ensemble_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return any(p.can_handle(request) for p in self._profilers)

    def profile(self, request: PipelineRequest) -> ProfilingResult[R]:
        contributions: dict[K, list[tuple[str, float, R]]] = {}
        for profiler, weight in zip(self._profilers, self._weights):
            if not profiler.can_handle(request):
                continue
            for rec in profiler.profile(request).recommendations:
                if self._filter_fn is not None and not self._filter_fn(rec):
                    continue
                contributions.setdefault(self._key_fn(rec), []).append((profiler.name, weight, rec))

        if self._mode == "max":
            recommendations = [
                max(items, key=lambda item: item[1] * item[2].confidence)[2]
                for items in contributions.values()
            ]
        else:
            assert self._build_fn is not None, "mode='weighted_sum' requires build_fn (checked in __init__)"
            build_fn = self._build_fn
            tallies: dict[K, Tally] = {}
            for candidate, items in contributions.items():
                tally = tallies.setdefault(candidate, Tally())
                for profiler_name, weight, rec in items:
                    tally.add(weight * rec.confidence, f"{profiler_name}: {rec.reasoning}")

            total = sum(t.total_weight for t in tallies.values()) or 1.0
            recommendations = [
                build_fn(candidate, round(tally.total_weight / total, 3), "; ".join(tally.reasons))
                for candidate, tally in tallies.items()
                if tally.total_weight > 0
            ]

        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        return ProfilingResult(request=request, recommendations=recommendations)


class ChainProfiler[R: Recommendation](Profiler[R]):
    """
    Tries profilers in order, returning the first result that has at least
    one recommendation. Falls back to the next profiler when one can't
    handle the request or produces no recommendations.

    Usage:
        profiler = ChainProfiler([LLMEngineProfiler(client), RuleBasedEngineProfiler()])
    """

    def __init__(self, profilers: list[Profiler[R]]) -> None:
        if not profilers:
            raise ValueError("ChainProfiler requires at least one profiler")
        self._profilers = profilers

    @property
    def name(self) -> str:
        return "chain_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return any(p.can_handle(request) for p in self._profilers)

    def profile(self, request: PipelineRequest) -> ProfilingResult[R]:
        last: ProfilingResult[R] | None = None
        for profiler in self._profilers:
            if not profiler.can_handle(request):
                continue
            result = profiler.profile(request)
            last = result
            if result.recommendations:
                return result
        return last if last is not None else ProfilingResult(request=request, recommendations=[])
