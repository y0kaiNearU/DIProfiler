import pytest

from core.profiler import Profiler
from models.models import (
    DatasetInfo,
    EngineType,
    FileFormat,
    FileSource,
    PipelineRequest,
    ProfilingResult,
    ResourceRecommendation,
)
from profilers.engine.cost_based_engine_profiler import CostBasedEngineProfiler, EngineCostRate

_GB = 1024 ** 3


class _StubResourceProfiler(Profiler):
    def __init__(self, cores, memory_bytes, can_handle=True):
        self._cores = cores
        self._memory_bytes = memory_bytes
        self._can_handle = can_handle

    @property
    def name(self):
        return "stub_resource_profiler"

    def can_handle(self, request):
        return self._can_handle

    def profile(self, request):
        rec = ResourceRecommendation(cores=self._cores, memory_bytes=self._memory_bytes, confidence=1.0, reasoning="stub")
        return ProfilingResult(request=request, recommendations=[rec])


def _req(available_engines=None, available_cores=8, available_memory_bytes=16 * _GB):
    return PipelineRequest(
        source=DatasetInfo(source=FileSource(path="x", format=FileFormat.CSV), size_bytes=1 * _GB),
        available_engines=available_engines if available_engines is not None else list(EngineType),
        available_cores=available_cores,
        available_memory_bytes=available_memory_bytes,
    )


class TestCostBasedEngineProfiler:
    def test_name(self):
        profiler = CostBasedEngineProfiler({}, resource_profiler=_StubResourceProfiler(4, 8 * _GB))
        assert profiler.name == "cost_based_engine_profiler"

    def test_can_handle_requires_resource_profiler_can_handle(self):
        profiler = CostBasedEngineProfiler(
            {EngineType.DUCKDB: EngineCostRate(0.0, 0.0)},
            resource_profiler=_StubResourceProfiler(4, 8 * _GB, can_handle=False),
        )
        assert not profiler.can_handle(_req())

    def test_can_handle_requires_at_least_one_engine_with_a_rate(self):
        profiler = CostBasedEngineProfiler({}, resource_profiler=_StubResourceProfiler(4, 8 * _GB))
        assert not profiler.can_handle(_req())

    def test_can_handle_true_when_rate_and_sizing_available(self):
        profiler = CostBasedEngineProfiler(
            {EngineType.DUCKDB: EngineCostRate(0.0, 0.0)},
            resource_profiler=_StubResourceProfiler(4, 8 * _GB),
        )
        assert profiler.can_handle(_req())

    def test_cheapest_engine_recommended_first(self):
        rates = {
            EngineType.DUCKDB: EngineCostRate(cost_per_core_hour=0.0, cost_per_gb_hour=0.0),
            EngineType.SPARK: EngineCostRate(cost_per_core_hour=0.05, cost_per_gb_hour=0.01),
        }
        profiler = CostBasedEngineProfiler(rates, resource_profiler=_StubResourceProfiler(4, 8 * _GB))
        result = profiler.profile(_req())
        assert result.best.engine == EngineType.DUCKDB

    def test_engines_without_rate_are_excluded(self):
        rates = {EngineType.DUCKDB: EngineCostRate(0.01, 0.01)}
        profiler = CostBasedEngineProfiler(rates, resource_profiler=_StubResourceProfiler(4, 8 * _GB))
        result = profiler.profile(_req(available_engines=[EngineType.DUCKDB, EngineType.SPARK]))
        engines = {r.engine for r in result.recommendations}
        assert engines == {EngineType.DUCKDB}

    def test_engines_not_in_available_engines_are_excluded(self):
        rates = {
            EngineType.DUCKDB: EngineCostRate(0.0, 0.0),
            EngineType.SPARK: EngineCostRate(0.05, 0.01),
        }
        profiler = CostBasedEngineProfiler(rates, resource_profiler=_StubResourceProfiler(4, 8 * _GB))
        result = profiler.profile(_req(available_engines=[EngineType.DUCKDB]))
        engines = {r.engine for r in result.recommendations}
        assert engines == {EngineType.DUCKDB}

    def test_confidences_sum_to_one(self):
        rates = {
            EngineType.DUCKDB: EngineCostRate(cost_per_core_hour=0.01, cost_per_gb_hour=0.01),
            EngineType.SPARK: EngineCostRate(cost_per_core_hour=0.05, cost_per_gb_hour=0.02),
        }
        profiler = CostBasedEngineProfiler(rates, resource_profiler=_StubResourceProfiler(4, 8 * _GB))
        result = profiler.profile(_req(available_engines=[EngineType.DUCKDB, EngineType.SPARK]))
        total = sum(r.confidence for r in result.recommendations)
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_recommendations_sorted_descending(self):
        rates = {
            EngineType.DUCKDB: EngineCostRate(cost_per_core_hour=0.0, cost_per_gb_hour=0.0),
            EngineType.SPARK: EngineCostRate(cost_per_core_hour=0.05, cost_per_gb_hour=0.02),
        }
        profiler = CostBasedEngineProfiler(rates, resource_profiler=_StubResourceProfiler(4, 8 * _GB))
        result = profiler.profile(_req(available_engines=[EngineType.DUCKDB, EngineType.SPARK]))
        confidences = [r.confidence for r in result.recommendations]
        assert confidences == sorted(confidences, reverse=True)

    def test_default_resource_profiler_is_rule_based(self):
        rates = {EngineType.DUCKDB: EngineCostRate(0.01, 0.01)}
        profiler = CostBasedEngineProfiler(rates)
        result = profiler.profile(_req(available_engines=[EngineType.DUCKDB]))
        assert result.best.engine == EngineType.DUCKDB
