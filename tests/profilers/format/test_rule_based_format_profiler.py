import pytest

from models.models import (
    DatasetInfo,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
    WriteMode,
)
from profilers.format.rule_based_format_profiler import (
    RuleBasedFormatProfiler,
    _append_write_mode_rule,
    _nested_schema_rule,
    _no_operations_rule,
    _operation_rule,
    _size_bytes_rule,
    _wide_schema_rule,
)

_GB = 1024 ** 3
_MB = 1024 ** 2


def _req(
    size_bytes=None,
    ops=None,
    num_columns=None,
    schema=None,
    destination=None,
    available_engines=None,
):
    return PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="x", format=FileFormat.CSV),
            size_bytes=size_bytes,
            num_columns=num_columns,
            schema=schema or {},
        ),
        operations=ops or [],
        destination=destination,
        available_engines=available_engines if available_engines is not None else list(EngineType),
    )


def _dest(write_mode=WriteMode.OVERWRITE, fmt=FileFormat.PARQUET):
    return DatasetInfo(source=FileSource(path="y", format=fmt, write_mode=write_mode))


class TestOperationRule:
    def test_ops_present_votes_parquet(self):
        fmt, _, _ = _operation_rule(_req(ops=[OperationType.AGGREGATE]))
        assert fmt == FileFormat.PARQUET

    def test_no_ops_returns_none(self):
        assert _operation_rule(_req()) is None


class TestNoOperationsRule:
    def test_no_ops_votes_csv(self):
        fmt, _, _ = _no_operations_rule(_req())
        assert fmt == FileFormat.CSV

    def test_ops_present_returns_none(self):
        assert _no_operations_rule(_req(ops=[OperationType.FILTER])) is None


class TestSizeBytesRule:
    def test_large_dataset_votes_parquet(self):
        fmt, _, _ = _size_bytes_rule(_req(size_bytes=5 * _GB))
        assert fmt == FileFormat.PARQUET

    def test_small_dataset_votes_csv(self):
        fmt, _, _ = _size_bytes_rule(_req(size_bytes=1 * _MB))
        assert fmt == FileFormat.CSV

    def test_medium_dataset_returns_none(self):
        assert _size_bytes_rule(_req(size_bytes=100 * _MB)) is None

    def test_no_size_returns_none(self):
        assert _size_bytes_rule(_req()) is None


class TestWideSchemaRule:
    def test_wide_with_selective_ops_votes_parquet(self):
        req = _req(num_columns=30, ops=[OperationType.FILTER])
        fmt, _, _ = _wide_schema_rule(req)
        assert fmt == FileFormat.PARQUET

    def test_wide_without_selective_ops_returns_none(self):
        req = _req(num_columns=30, ops=[OperationType.SORT])
        assert _wide_schema_rule(req) is None

    def test_narrow_schema_returns_none(self):
        req = _req(num_columns=5, ops=[OperationType.FILTER])
        assert _wide_schema_rule(req) is None

    def test_no_num_columns_returns_none(self):
        assert _wide_schema_rule(_req(ops=[OperationType.FILTER])) is None


class TestNestedSchemaRule:
    def test_nested_type_votes_json(self):
        req = _req(schema={"payload": "struct<a:int,b:string>"})
        fmt, _, _ = _nested_schema_rule(req)
        assert fmt == FileFormat.JSON

    def test_flat_schema_returns_none(self):
        req = _req(schema={"amount": "float64", "name": "string"})
        assert _nested_schema_rule(req) is None

    def test_empty_schema_returns_none(self):
        assert _nested_schema_rule(_req()) is None


class TestAppendWriteModeRule:
    def test_append_with_spark_votes_delta(self):
        req = _req(destination=_dest(write_mode=WriteMode.APPEND), available_engines=[EngineType.SPARK])
        fmt, _, _ = _append_write_mode_rule(req)
        assert fmt == FileFormat.DELTA

    def test_append_without_spark_returns_none(self):
        req = _req(destination=_dest(write_mode=WriteMode.APPEND), available_engines=[EngineType.DUCKDB])
        assert _append_write_mode_rule(req) is None

    def test_overwrite_mode_returns_none(self):
        req = _req(destination=_dest(write_mode=WriteMode.OVERWRITE), available_engines=[EngineType.SPARK])
        assert _append_write_mode_rule(req) is None

    def test_no_destination_returns_none(self):
        assert _append_write_mode_rule(_req(available_engines=[EngineType.SPARK])) is None


class TestRuleBasedFormatProfiler:
    def setup_method(self):
        self.profiler = RuleBasedFormatProfiler()

    def test_name(self):
        assert self.profiler.name == "rule_based_format_profiler"

    def test_can_handle_always_true(self):
        assert self.profiler.can_handle(_req())

    def test_large_dataset_with_ops_recommends_parquet(self):
        req = _req(size_bytes=5 * _GB, ops=[OperationType.AGGREGATE])
        result = self.profiler.profile(req)
        assert result.best.format == FileFormat.PARQUET

    def test_no_ops_small_dataset_recommends_csv(self):
        req = _req(size_bytes=1 * _MB)
        result = self.profiler.profile(req)
        assert result.best.format == FileFormat.CSV

    def test_nested_schema_recommends_json(self):
        req = _req(schema={"payload": "map<string,string>"})
        result = self.profiler.profile(req)
        assert result.best.format == FileFormat.JSON

    def test_recommendations_sorted_descending(self):
        req = _req(size_bytes=5 * _GB, ops=[OperationType.AGGREGATE])
        result = self.profiler.profile(req)
        confidences = [r.confidence for r in result.recommendations]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidences_sum_to_one(self):
        req = _req(size_bytes=5 * _GB, ops=[OperationType.AGGREGATE])
        result = self.profiler.profile(req)
        total = sum(r.confidence for r in result.recommendations)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_no_rules_fire_returns_empty(self):
        profiler = RuleBasedFormatProfiler(rules=[])
        result = profiler.profile(_req())
        assert result.recommendations == []

    def test_custom_rules_list(self):
        called = []

        def my_rule(req):
            called.append(True)
            return FileFormat.ORC, 1.0, "custom rule"

        profiler = RuleBasedFormatProfiler(rules=[my_rule])
        result = profiler.profile(_req())
        assert called
        assert result.best.format == FileFormat.ORC
