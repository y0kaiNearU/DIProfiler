import pytest

from core.registry import ProfilerRegistry
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
        self.profile_call_count = 0

    @property
    def name(self):
        return self._name

    def can_handle(self, request):
        return self._handles

    def profile(self, request):
        self.profile_call_count += 1
        return ProfilingResult(request=request, recommendations=list(self._recommendations))


def _req():
    return PipelineRequest(
        source=DatasetInfo(source=FileSource(path="x.csv", format=FileFormat.CSV))
    )


def _rec(engine=EngineType.DUCKDB, confidence=0.9):
    return EngineRecommendation(engine=engine, confidence=confidence, reasoning="test")


class TestProfilerRegistryRegister:
    def test_register_single_profiler(self):
        registry = ProfilerRegistry()
        p = _StubProfiler("p1")
        registry.register(p)
        assert "p1" in registry.names

    def test_register_multiple_profilers(self):
        registry = ProfilerRegistry()
        registry.register(_StubProfiler("a"))
        registry.register(_StubProfiler("b"))
        assert set(registry.names) == {"a", "b"}

    def test_duplicate_name_raises(self):
        registry = ProfilerRegistry()
        registry.register(_StubProfiler("dup"))
        with pytest.raises(ValueError, match="dup"):
            registry.register(_StubProfiler("dup"))

    def test_unregister_removes_profiler(self):
        registry = ProfilerRegistry()
        registry.register(_StubProfiler("p"))
        registry.unregister("p")
        assert "p" not in registry.names

    def test_unregister_nonexistent_is_noop(self):
        registry = ProfilerRegistry()
        registry.unregister("does_not_exist")  # should not raise


class TestProfilerRegistryRun:
    def test_run_calls_all_eligible_profilers(self):
        registry = ProfilerRegistry()
        p1 = _StubProfiler("p1")
        p2 = _StubProfiler("p2")
        registry.register(p1)
        registry.register(p2)

        registry.run(_req())

        assert p1.profile_call_count == 1
        assert p2.profile_call_count == 1

    def test_run_skips_profilers_that_cannot_handle(self):
        registry = ProfilerRegistry()
        eligible = _StubProfiler("eligible", handles=True)
        skipped = _StubProfiler("skipped", handles=False)
        registry.register(eligible)
        registry.register(skipped)

        registry.run(_req())

        assert eligible.profile_call_count == 1
        assert skipped.profile_call_count == 0

    def test_run_returns_results_for_each_eligible_profiler(self):
        registry = ProfilerRegistry()
        registry.register(_StubProfiler("p1"))
        registry.register(_StubProfiler("p2"))

        results = registry.run(_req())

        assert len(results) == 2

    def test_run_empty_registry_returns_empty(self):
        registry = ProfilerRegistry()
        assert registry.run(_req()) == []

    def test_run_results_contain_recommendations(self):
        rec = _rec()
        registry = ProfilerRegistry()
        registry.register(_StubProfiler("p", recommendations=[rec]))

        results = registry.run(_req())

        assert results[0].recommendations[0] == rec


class TestProfilerRegistryGet:
    def test_get_known_profiler(self):
        registry = ProfilerRegistry()
        p = _StubProfiler("mine")
        registry.register(p)
        assert registry.get("mine") is p

    def test_get_unknown_raises(self):
        registry = ProfilerRegistry()
        with pytest.raises(KeyError):
            registry.get("unknown")

    def test_run_one_returns_result(self):
        rec = _rec(engine=EngineType.SPARK)
        registry = ProfilerRegistry()
        registry.register(_StubProfiler("p", recommendations=[rec]))

        result = registry.run_one("p", _req())

        assert result.recommendations[0].engine == EngineType.SPARK

    def test_run_one_unknown_raises(self):
        registry = ProfilerRegistry()
        with pytest.raises(KeyError):
            registry.run_one("ghost", _req())
