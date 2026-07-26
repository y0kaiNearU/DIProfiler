import pytest

from engine_selection.pipeline import DIProfiler
from models.models import (
    DatasetInfo,
    EngineRecommendation,
    EngineType,
    FileFormat,
    FileSource,
    PipelineRequest,
    ProfilingResult,
)


class _StubProfiler:
    def __init__(self, name, handles=True, recommendations=None):
        self._name = name
        self._handles = handles
        self._recommendations = recommendations or []

    @property
    def name(self):
        return self._name

    def can_handle(self, request):
        return self._handles

    def profile(self, request):
        return ProfilingResult(request=request, recommendations=list(self._recommendations))


def _req(fmt=FileFormat.CSV):
    return PipelineRequest(source=DatasetInfo(source=FileSource(path="x.csv", format=fmt)))


def _rec(engine, confidence, reasoning="test"):
    return EngineRecommendation(engine=engine, confidence=confidence, reasoning=reasoning)


class TestRecommend:
    def test_no_profiler_can_handle_raises(self):
        profiler = DIProfiler(
            profilers=[_StubProfiler("p", handles=False)],
            available_engines=[EngineType.DUCKDB],
        )
        with pytest.raises(RuntimeError, match="No profiler could handle"):
            profiler.recommend(_req())

    def test_single_profiler_recommendation_passes_through(self):
        stub = _StubProfiler("p", recommendations=[_rec(EngineType.DUCKDB, 0.9)])
        profiler = DIProfiler(profilers=[stub], available_engines=[EngineType.DUCKDB])
        result = profiler.recommend(_req())
        assert result.best.engine == EngineType.DUCKDB
        assert result.best.confidence == 0.9

    def test_takes_max_confidence_across_profilers(self):
        low = _StubProfiler("low", recommendations=[_rec(EngineType.DUCKDB, 0.3, "low says duckdb")])
        high = _StubProfiler("high", recommendations=[_rec(EngineType.DUCKDB, 0.8, "high says duckdb")])
        profiler = DIProfiler(profilers=[low, high], available_engines=[EngineType.DUCKDB])
        result = profiler.recommend(_req())
        assert result.best.confidence == 0.8
        assert result.best.reasoning == "high says duckdb"

    def test_engines_without_required_capability_are_filtered_out(self):
        stub = _StubProfiler("p", recommendations=[_rec(EngineType.DUCKDB, 0.9)])
        profiler = DIProfiler(profilers=[stub], available_engines=[EngineType.DUCKDB])
        with pytest.raises(RuntimeError, match="No engine can handle"):
            profiler.recommend(_req(fmt=FileFormat.ORC))

    def test_multiple_engines_recommended_and_sorted(self):
        stub = _StubProfiler("p", recommendations=[
            _rec(EngineType.DUCKDB, 0.4),
            _rec(EngineType.SPARK, 0.9),
        ])
        profiler = DIProfiler(profilers=[stub], available_engines=[EngineType.DUCKDB, EngineType.SPARK])
        result = profiler.recommend(_req())
        assert [r.engine for r in result.recommendations] == [EngineType.SPARK, EngineType.DUCKDB]
