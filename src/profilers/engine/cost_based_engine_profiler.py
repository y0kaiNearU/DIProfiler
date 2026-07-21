from __future__ import annotations

from dataclasses import dataclass

from core.profiler import Profiler
from models.models import EngineRecommendation, EngineType, PipelineRequest, ProfilingResult, ResourceRecommendation
from profilers.resource.rule_based_resource_profiler import RuleBasedResourceProfiler

_GB = 1024 ** 3
_EPSILON = 1e-9


@dataclass
class EngineCostRate:
    cost_per_core_hour: float
    cost_per_gb_hour: float


class CostBasedEngineProfiler(Profiler[EngineRecommendation]):
    """
    Recommends the cheapest available engine by estimating an hourly cost at
    the resource allocation a resource profiler (RuleBasedResourceProfiler by
    default) sizes for the request:

        hourly_cost = cores * cost_per_core_hour + (memory_bytes / 1GB) * cost_per_gb_hour

    This is a cost-per-hour-of-resources estimate, not a total job cost — it
    doesn't model engine throughput/runtime, so it can't weigh "$3/hr on a
    slow engine" against "$10/hr on a fast one" in terms of total spend.

    cost_rates has no sensible auto-detected default (there's no way to infer
    what your Spark cluster or warehouse charges per hour from the local
    machine) — it must be supplied by the caller for every engine they want
    considered; engines missing from cost_rates are never recommended.

    Usage:
        rates = {
            EngineType.DUCKDB: EngineCostRate(cost_per_core_hour=0.0, cost_per_gb_hour=0.0),  # local, free
            EngineType.SPARK: EngineCostRate(cost_per_core_hour=0.05, cost_per_gb_hour=0.01),  # cluster
        }
        profiler = CostBasedEngineProfiler(rates)
    """

    def __init__(
        self,
        cost_rates: dict[EngineType, EngineCostRate],
        resource_profiler: Profiler[ResourceRecommendation] | None = None,
    ) -> None:
        self._cost_rates = cost_rates
        self._resource_profiler = resource_profiler or RuleBasedResourceProfiler()

    @property
    def name(self) -> str:
        return "cost_based_engine_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        if not self._resource_profiler.can_handle(request):
            return False
        return any(engine in self._cost_rates for engine in request.available_engines)

    def profile(self, request: PipelineRequest) -> ProfilingResult[EngineRecommendation]:
        sizing = self._resource_profiler.profile(request).best
        if sizing is None:
            return ProfilingResult(request=request, recommendations=[])
        memory_gb = sizing.memory_bytes / _GB

        costs: dict[EngineType, float] = {}
        for engine in request.available_engines:
            rate = self._cost_rates.get(engine)
            if rate is None:
                continue
            costs[engine] = sizing.cores * rate.cost_per_core_hour + memory_gb * rate.cost_per_gb_hour

        scores = {engine: 1.0 / max(cost, _EPSILON) for engine, cost in costs.items()}
        total = sum(scores.values()) or 1.0

        recommendations = [
            EngineRecommendation(
                engine=engine,
                confidence=round(scores[engine] / total, 3),
                reasoning=f"estimated ${costs[engine]:.4f}/hr for {sizing.cores} cores / {memory_gb:.1f} GB",
            )
            for engine in costs
        ]
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return ProfilingResult(request=request, recommendations=recommendations)
