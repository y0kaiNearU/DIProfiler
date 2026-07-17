from __future__ import annotations

from core.profiler import Profiler
from models.models import EngineRecommendation, EngineType, PipelineRequest, ProfilingResult
from profilers.common.voting import Tally


class EnsembleProfiler(Profiler):
    """
    Combines multiple profilers by weighted confidence voting.

    Each sub-profiler that can_handle the request contributes its
    recommendations scaled by its weight; confidences are summed per engine
    and renormalized to sum to 1.0.

    Usage:
        profiler = EnsembleProfiler([RuleBasedEngineProfiler(), MLEngineProfiler(model)])
        profiler = EnsembleProfiler([rule_profiler, ml_profiler], weights=[0.4, 0.6])
    """

    def __init__(self, profilers: list[Profiler], weights: list[float] | None = None) -> None:
        if weights is not None and len(weights) != len(profilers):
            raise ValueError("weights must have the same length as profilers")
        self._profilers = profilers
        self._weights = weights if weights is not None else [1.0] * len(profilers)

    @property
    def name(self) -> str:
        return "ensemble_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return any(p.can_handle(request) for p in self._profilers)

    def profile(self, request: PipelineRequest) -> ProfilingResult:
        tallies: dict[EngineType, Tally] = {}
        for profiler, weight in zip(self._profilers, self._weights):
            if not profiler.can_handle(request):
                continue
            for rec in profiler.profile(request).recommendations:
                tally = tallies.setdefault(rec.engine, Tally())
                tally.add(weight * rec.confidence, f"{profiler.name}: {rec.reasoning}")

        total = sum(t.total_weight for t in tallies.values()) or 1.0
        recommendations = [
            EngineRecommendation(
                engine=engine,
                confidence=round(tally.total_weight / total, 3),
                reasoning="; ".join(tally.reasons),
            )
            for engine, tally in tallies.items()
            if tally.total_weight > 0
        ]
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return ProfilingResult(request=request, recommendations=recommendations)


class ChainProfiler(Profiler):
    """
    Tries profilers in order, returning the first result that has at least
    one recommendation. Falls back to the next profiler when one can't
    handle the request or produces no recommendations.

    Usage:
        profiler = ChainProfiler([LLMEngineProfiler(client), RuleBasedEngineProfiler()])
    """

    def __init__(self, profilers: list[Profiler]) -> None:
        if not profilers:
            raise ValueError("ChainProfiler requires at least one profiler")
        self._profilers = profilers

    @property
    def name(self) -> str:
        return "chain_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return any(p.can_handle(request) for p in self._profilers)

    def profile(self, request: PipelineRequest) -> ProfilingResult:
        last: ProfilingResult | None = None
        for profiler in self._profilers:
            if not profiler.can_handle(request):
                continue
            result = profiler.profile(request)
            last = result
            if result.recommendations:
                return result
        return last if last is not None else ProfilingResult(request=request, recommendations=[])
