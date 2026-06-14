import pytest
from unittest.mock import MagicMock

from models.models import (
    DatasetInfo,
    EngineRecommendation,
    EngineType,
    FileFormat,
    FileSource,
    PipelineRequest,
    ProfilingResult,
)
from profilers.prefetching_profiler import PrefetchingProfiler


def _make_source(size_bytes=None, row_count=None):
    return DatasetInfo(
        source=FileSource(path="x.csv", format=FileFormat.CSV),
        size_bytes=size_bytes,
        row_count=row_count,
    )


def _req(size_bytes=None, row_count=None):
    return PipelineRequest(
        source=_make_source(size_bytes=size_bytes, row_count=row_count),
        available_engines=list(EngineType),
    )


def _enriched_info(size_bytes=500 * 1024 ** 2, row_count=1_000_000):
    return DatasetInfo(
        source=FileSource(path="x.csv", format=FileFormat.CSV),
        size_bytes=size_bytes,
        row_count=row_count,
        num_columns=10,
    )


def test_can_handle_when_size_missing():
    profiler = PrefetchingProfiler(prefetch_fn=lambda r: _enriched_info())
    assert profiler.can_handle(_req(row_count=100))


def test_can_handle_when_row_count_missing():
    profiler = PrefetchingProfiler(prefetch_fn=lambda r: _enriched_info())
    assert profiler.can_handle(_req(size_bytes=1000))


def test_can_handle_when_both_missing():
    profiler = PrefetchingProfiler(prefetch_fn=lambda r: _enriched_info())
    assert profiler.can_handle(_req())


def test_cannot_handle_when_both_present():
    profiler = PrefetchingProfiler(prefetch_fn=lambda r: _enriched_info())
    assert not profiler.can_handle(_req(size_bytes=1000, row_count=100))


def test_name():
    assert PrefetchingProfiler(prefetch_fn=lambda r: _enriched_info()).name == "prefetching_profiler"


def test_prefetch_fn_called_with_original_request():
    received = []

    def prefetch(req):
        received.append(req)
        return _enriched_info()

    delegate = MagicMock()
    delegate.profile.return_value = ProfilingResult(request=_req(), recommendations=[])

    profiler = PrefetchingProfiler(prefetch_fn=prefetch, delegate=delegate)
    original = _req()
    profiler.profile(original)

    assert received[0] is original


def test_delegate_receives_enriched_request():
    enriched = _enriched_info(size_bytes=999, row_count=777)

    received = []

    class CapturingDelegate:
        name = "capturing"
        def can_handle(self, req): return True
        def profile(self, req):
            received.append(req)
            return ProfilingResult(request=req, recommendations=[])

    profiler = PrefetchingProfiler(prefetch_fn=lambda r: enriched, delegate=CapturingDelegate())
    profiler.profile(_req())

    assert received[0].source.size_bytes == 999
    assert received[0].source.row_count == 777


def test_profile_returns_delegate_result():
    expected_rec = EngineRecommendation(engine=EngineType.DUCKDB, confidence=1.0, reasoning="test")

    def prefetch(req):
        return _enriched_info()

    delegate = MagicMock()
    req = _req()
    delegate.profile.return_value = ProfilingResult(request=req, recommendations=[expected_rec])

    profiler = PrefetchingProfiler(prefetch_fn=prefetch, delegate=delegate)
    result = profiler.profile(req)

    assert result.recommendations[0] == expected_rec


def test_default_delegate_is_rule_based():
    from profilers.engine_profiler import RuleBasedEngineProfiler

    profiler = PrefetchingProfiler(prefetch_fn=lambda r: _enriched_info(size_bytes=100 * 1024 ** 2))
    result = profiler.profile(_req())
    # With 100MB CSV, rule-based picks DuckDB
    assert result.best is not None
    assert result.best.engine == EngineType.DUCKDB


def test_default_delegate_is_created_lazily():
    # _delegate should be None before first profile call
    profiler = PrefetchingProfiler(prefetch_fn=lambda r: _enriched_info())
    assert profiler._delegate is None
    profiler.profile(_req())
    assert profiler._delegate is not None
