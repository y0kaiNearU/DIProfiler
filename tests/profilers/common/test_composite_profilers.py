import pytest

from core.profiler import Profiler
from models.models import (
    DatasetInfo,
    EngineRecommendation,
    EngineType,
    FileFormat,
    FileSource,
    PipelineRequest,
    ProfilingResult,
)
from profilers.common.composite import ChainProfiler, EnsembleProfiler


def _req():
    return PipelineRequest(source=DatasetInfo(source=FileSource(path="x", format=FileFormat.CSV)))


def _engine_key(rec):
    return rec.engine


def _engine_build(engine, confidence, reasoning):
    return EngineRecommendation(engine=engine, confidence=confidence, reasoning=reasoning)


class _StubProfiler(Profiler):
    def __init__(self, name, can_handle=True, recommendations=None):
        self._name = name
        self._can_handle = can_handle
        self._recommendations = recommendations or []

    @property
    def name(self):
        return self._name

    def can_handle(self, request):
        return self._can_handle

    def profile(self, request):
        return ProfilingResult(request=request, recommendations=list(self._recommendations))


class TestEnsembleProfiler:
    def test_name(self):
        profiler = EnsembleProfiler([_StubProfiler("a")], key_fn=_engine_key, build_fn=_engine_build)
        assert profiler.name == "ensemble_profiler"

    def test_can_handle_true_if_any_subprofiler_can_handle(self):
        profiler = EnsembleProfiler(
            [_StubProfiler("a", can_handle=False), _StubProfiler("b", can_handle=True)],
            key_fn=_engine_key,
            build_fn=_engine_build,
        )
        assert profiler.can_handle(_req())

    def test_can_handle_false_if_none_can_handle(self):
        profiler = EnsembleProfiler([_StubProfiler("a", can_handle=False)], key_fn=_engine_key, build_fn=_engine_build)
        assert not profiler.can_handle(_req())

    def test_skips_profilers_that_cannot_handle(self):
        a = _StubProfiler("a", can_handle=False, recommendations=[
            EngineRecommendation(engine=EngineType.SPARK, confidence=1.0, reasoning="a"),
        ])
        b = _StubProfiler("b", can_handle=True, recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=1.0, reasoning="b"),
        ])
        profiler = EnsembleProfiler([a, b], key_fn=_engine_key, build_fn=_engine_build)
        result = profiler.profile(_req())
        engines = {r.engine for r in result.recommendations}
        assert engines == {EngineType.DUCKDB}

    def test_weighted_confidence_combines_across_profilers(self):
        a = _StubProfiler("a", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=1.0, reasoning="a likes duckdb"),
        ])
        b = _StubProfiler("b", recommendations=[
            EngineRecommendation(engine=EngineType.SPARK, confidence=1.0, reasoning="b likes spark"),
        ])
        profiler = EnsembleProfiler([a, b], key_fn=_engine_key, build_fn=_engine_build, weights=[0.25, 0.75])
        result = profiler.profile(_req())
        by_engine = {r.engine: r.confidence for r in result.recommendations}
        assert by_engine[EngineType.SPARK] > by_engine[EngineType.DUCKDB]
        assert by_engine[EngineType.DUCKDB] == pytest.approx(0.25, abs=1e-6)
        assert by_engine[EngineType.SPARK] == pytest.approx(0.75, abs=1e-6)

    def test_recommendations_sorted_descending(self):
        a = _StubProfiler("a", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.3, reasoning="a"),
            EngineRecommendation(engine=EngineType.SPARK, confidence=0.9, reasoning="a"),
        ])
        profiler = EnsembleProfiler([a], key_fn=_engine_key, build_fn=_engine_build)
        result = profiler.profile(_req())
        confidences = [r.confidence for r in result.recommendations]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidences_sum_to_one(self):
        a = _StubProfiler("a", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.3, reasoning="a"),
            EngineRecommendation(engine=EngineType.SPARK, confidence=0.9, reasoning="a"),
        ])
        profiler = EnsembleProfiler([a], key_fn=_engine_key, build_fn=_engine_build)
        result = profiler.profile(_req())
        total = sum(r.confidence for r in result.recommendations)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_weights_length_raises(self):
        with pytest.raises(ValueError):
            EnsembleProfiler(
                [_StubProfiler("a"), _StubProfiler("b")],
                key_fn=_engine_key,
                build_fn=_engine_build,
                weights=[1.0],
            )

    def test_no_recommendations_returns_empty(self):
        profiler = EnsembleProfiler(
            [_StubProfiler("a", recommendations=[])], key_fn=_engine_key, build_fn=_engine_build
        )
        result = profiler.profile(_req())
        assert result.recommendations == []

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            EnsembleProfiler([_StubProfiler("a")], key_fn=_engine_key, mode="median")

    def test_weighted_sum_without_build_fn_raises(self):
        with pytest.raises(ValueError):
            EnsembleProfiler([_StubProfiler("a")], key_fn=_engine_key)

    def test_max_mode_picks_single_highest_recommendation_unchanged(self):
        a = _StubProfiler("a", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.4, reasoning="a says duckdb"),
        ])
        b = _StubProfiler("b", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.9, reasoning="b says duckdb"),
        ])
        profiler = EnsembleProfiler([a, b], key_fn=_engine_key, mode="max")
        result = profiler.profile(_req())
        assert len(result.recommendations) == 1
        assert result.recommendations[0].confidence == 0.9
        assert result.recommendations[0].reasoning == "b says duckdb"

    def test_max_mode_keeps_engines_independent(self):
        a = _StubProfiler("a", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.4, reasoning="a"),
            EngineRecommendation(engine=EngineType.SPARK, confidence=0.7, reasoning="a"),
        ])
        profiler = EnsembleProfiler([a], key_fn=_engine_key, mode="max")
        result = profiler.profile(_req())
        by_engine = {r.engine: r.confidence for r in result.recommendations}
        assert by_engine == {EngineType.DUCKDB: 0.4, EngineType.SPARK: 0.7}

    def test_filter_fn_excludes_recommendations(self):
        a = _StubProfiler("a", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.4, reasoning="a"),
            EngineRecommendation(engine=EngineType.SPARK, confidence=0.7, reasoning="a"),
        ])
        profiler = EnsembleProfiler(
            [a], key_fn=_engine_key, build_fn=_engine_build, filter_fn=lambda rec: rec.engine == EngineType.SPARK,
        )
        result = profiler.profile(_req())
        engines = {r.engine for r in result.recommendations}
        assert engines == {EngineType.SPARK}


class TestChainProfiler:
    def test_name(self):
        assert ChainProfiler([_StubProfiler("a")]).name == "chain_profiler"

    def test_empty_profiler_list_raises(self):
        with pytest.raises(ValueError):
            ChainProfiler([])

    def test_can_handle_true_if_any_subprofiler_can_handle(self):
        profiler = ChainProfiler([_StubProfiler("a", can_handle=False), _StubProfiler("b", can_handle=True)])
        assert profiler.can_handle(_req())

    def test_returns_first_nonempty_result(self):
        first = _StubProfiler("first", recommendations=[
            EngineRecommendation(engine=EngineType.SPARK, confidence=0.9, reasoning="first"),
        ])
        second = _StubProfiler("second", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.5, reasoning="second"),
        ])
        profiler = ChainProfiler([first, second])
        result = profiler.profile(_req())
        assert result.recommendations[0].engine == EngineType.SPARK

    def test_falls_back_when_first_has_no_recommendations(self):
        first = _StubProfiler("first", recommendations=[])
        second = _StubProfiler("second", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.5, reasoning="second"),
        ])
        profiler = ChainProfiler([first, second])
        result = profiler.profile(_req())
        assert result.recommendations[0].engine == EngineType.DUCKDB

    def test_falls_back_when_first_cannot_handle(self):
        first = _StubProfiler("first", can_handle=False, recommendations=[
            EngineRecommendation(engine=EngineType.SPARK, confidence=0.9, reasoning="first"),
        ])
        second = _StubProfiler("second", recommendations=[
            EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.5, reasoning="second"),
        ])
        profiler = ChainProfiler([first, second])
        result = profiler.profile(_req())
        assert result.recommendations[0].engine == EngineType.DUCKDB

    def test_returns_last_result_if_all_empty(self):
        first = _StubProfiler("first", recommendations=[])
        second = _StubProfiler("second", recommendations=[])
        profiler = ChainProfiler([first, second])
        result = profiler.profile(_req())
        assert result.recommendations == []

    def test_returns_empty_result_if_none_can_handle(self):
        first = _StubProfiler("first", can_handle=False)
        profiler = ChainProfiler([first])
        result = profiler.profile(_req())
        assert result.recommendations == []
