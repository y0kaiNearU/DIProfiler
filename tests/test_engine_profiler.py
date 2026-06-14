import pytest

from models.models import (
    DatasetInfo,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
)
from profilers.engine_profiler import (
    DEFAULT_RULES,
    RuleBasedEngineProfiler,
    _datafusion_row_count_rule,
    _datafusion_size_rule,
    _format_rule,
    _operation_rule,
    _required_engine_rule,
    _row_count_rule,
    _size_bytes_rule,
)

_GB = 1024 ** 3
_MB = 1024 ** 2


def _req(
    size_bytes=None,
    row_count=None,
    fmt=FileFormat.CSV,
    ops=None,
    required_engine=None,
    available_engines=None,
):
    return PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="x", format=fmt),
            size_bytes=size_bytes,
            row_count=row_count,
        ),
        operations=ops or [],
        required_engine=required_engine,
        available_engines=available_engines if available_engines is not None else list(EngineType),
    )


class TestRequiredEngineRule:
    def test_returns_vote_when_set(self):
        req = _req(required_engine=EngineType.SPARK)
        vote = _required_engine_rule(req)
        assert vote is not None
        engine, weight, _ = vote
        assert engine == EngineType.SPARK
        assert weight == 1.0

    def test_returns_none_when_not_set(self):
        assert _required_engine_rule(_req()) is None


class TestSizeBytesRule:
    def test_large_dataset_votes_spark(self):
        engine, _, _ = _size_bytes_rule(_req(size_bytes=20 * _GB))
        assert engine == EngineType.SPARK

    def test_small_dataset_votes_duckdb(self):
        engine, _, _ = _size_bytes_rule(_req(size_bytes=500 * _MB))
        assert engine == EngineType.DUCKDB

    def test_medium_dataset_returns_none(self):
        assert _size_bytes_rule(_req(size_bytes=5 * _GB)) is None

    def test_no_size_returns_none(self):
        assert _size_bytes_rule(_req()) is None


class TestDataFusionSizeRule:
    def test_small_dataset_votes_datafusion(self):
        engine, _, _ = _datafusion_size_rule(_req(size_bytes=200 * _MB))
        assert engine == EngineType.DATAFUSION

    def test_large_dataset_returns_none(self):
        assert _datafusion_size_rule(_req(size_bytes=5 * _GB)) is None

    def test_no_size_returns_none(self):
        assert _datafusion_size_rule(_req()) is None


class TestRowCountRule:
    def test_huge_row_count_votes_spark(self):
        engine, _, _ = _row_count_rule(_req(row_count=200_000_000))
        assert engine == EngineType.SPARK

    def test_small_row_count_votes_duckdb(self):
        engine, _, _ = _row_count_rule(_req(row_count=1_000))
        assert engine == EngineType.DUCKDB

    def test_no_row_count_returns_none(self):
        assert _row_count_rule(_req()) is None


class TestDataFusionRowCountRule:
    def test_small_row_count_votes_datafusion(self):
        engine, _, _ = _datafusion_row_count_rule(_req(row_count=50_000))
        assert engine == EngineType.DATAFUSION

    def test_huge_row_count_returns_none(self):
        assert _datafusion_row_count_rule(_req(row_count=200_000_000)) is None

    def test_no_row_count_returns_none(self):
        assert _datafusion_row_count_rule(_req()) is None


class TestFormatRule:
    def test_orc_votes_spark(self):
        engine, _, _ = _format_rule(_req(fmt=FileFormat.ORC))
        assert engine == EngineType.SPARK

    def test_delta_votes_spark(self):
        engine, _, _ = _format_rule(_req(fmt=FileFormat.DELTA))
        assert engine == EngineType.SPARK

    def test_parquet_votes_datafusion(self):
        engine, _, _ = _format_rule(_req(fmt=FileFormat.PARQUET))
        assert engine == EngineType.DATAFUSION

    def test_csv_votes_duckdb(self):
        engine, _, _ = _format_rule(_req(fmt=FileFormat.CSV))
        assert engine == EngineType.DUCKDB

    def test_json_votes_duckdb(self):
        engine, _, _ = _format_rule(_req(fmt=FileFormat.JSON))
        assert engine == EngineType.DUCKDB

    def test_iceberg_returns_none(self):
        assert _format_rule(_req(fmt=FileFormat.ICEBERG)) is None


class TestOperationRule:
    def test_heavy_ops_on_large_data_votes_spark(self):
        req = _req(size_bytes=5 * _GB, ops=[OperationType.JOIN, OperationType.WINDOW])
        engine, _, _ = _operation_rule(req)
        assert engine == EngineType.SPARK

    def test_heavy_ops_on_small_data_returns_none(self):
        req = _req(size_bytes=100 * _MB, ops=[OperationType.JOIN])
        assert _operation_rule(req) is None

    def test_no_ops_returns_none(self):
        assert _operation_rule(_req(size_bytes=5 * _GB)) is None


class TestRuleBasedEngineProfiler:
    def setup_method(self):
        self.profiler = RuleBasedEngineProfiler()

    def test_name(self):
        assert self.profiler.name == "rule_based_engine_profiler"

    def test_can_handle_always_true(self):
        assert self.profiler.can_handle(_req())

    def test_large_orc_recommends_spark(self):
        req = _req(size_bytes=50 * _GB, fmt=FileFormat.ORC)
        result = self.profiler.profile(req)
        assert result.best.engine == EngineType.SPARK

    def test_small_csv_recommends_duckdb(self):
        req = _req(size_bytes=100 * _MB, fmt=FileFormat.CSV, row_count=500_000)
        result = self.profiler.profile(req)
        assert result.best.engine == EngineType.DUCKDB

    def test_medium_parquet_recommends_datafusion(self):
        req = _req(size_bytes=800 * _MB, fmt=FileFormat.PARQUET, row_count=4_000_000)
        result = self.profiler.profile(req)
        assert result.best.engine == EngineType.DATAFUSION

    def test_recommendations_sorted_descending(self):
        req = _req(size_bytes=100 * _MB, fmt=FileFormat.CSV, row_count=500_000)
        result = self.profiler.profile(req)
        confidences = [r.confidence for r in result.recommendations]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidences_sum_to_one(self):
        req = _req(size_bytes=100 * _MB, fmt=FileFormat.CSV, row_count=500_000)
        result = self.profiler.profile(req)
        total = sum(r.confidence for r in result.recommendations)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_available_engines_filter(self):
        req = _req(size_bytes=100 * _MB, available_engines=[EngineType.SPARK])
        result = self.profiler.profile(req)
        for rec in result.recommendations:
            assert rec.engine == EngineType.SPARK

    def test_no_rules_fire_returns_empty(self):
        profiler = RuleBasedEngineProfiler(rules=[])
        result = profiler.profile(_req())
        assert result.recommendations == []

    def test_custom_rules_list(self):
        called = []

        def my_rule(req):
            called.append(True)
            return EngineType.DUCKDB, 1.0, "custom rule"

        profiler = RuleBasedEngineProfiler(rules=[my_rule])
        result = profiler.profile(_req())
        assert called
        assert result.best.engine == EngineType.DUCKDB
