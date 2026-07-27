from __future__ import annotations

from core.profiler import Profiler
from models.models import (
    EngineType,
    FileFormat,
    FileSource,
    FormatRecommendation,
    OperationType,
    PipelineRequest,
    ProfilingResult,
    WriteMode,
)
from profilers.common.voting import Rule as _Rule, Vote as _Vote, aggregate_votes

Vote = _Vote[FileFormat]
Rule = _Rule[FileFormat]

_GB = 1024 ** 3
_MB = 1024 ** 2
_LARGE_DATASET_BYTES = 1 * _GB
_SMALL_DATASET_BYTES = 10 * _MB
_WIDE_SCHEMA_COLUMNS = 20
_NESTED_TYPE_HINTS = ("struct", "array", "map", "object", "list", "json")


def _operation_rule(req: PipelineRequest) -> Vote | None:
    ops = req.operations
    if not ops:
        return None
    return (
        FileFormat.PARQUET,
        0.5,
        f"operations {[o.value for o in ops]} benefit from Parquet's columnar layout and predicate pushdown",
    )


def _no_operations_rule(req: PipelineRequest) -> Vote | None:
    if req.operations:
        return None
    return FileFormat.CSV, 0.3, "no transformations planned; CSV is simple to produce and inspect"


def _size_bytes_rule(req: PipelineRequest) -> Vote | None:
    size = req.source.size_bytes
    if size is None:
        return None
    if size >= _LARGE_DATASET_BYTES:
        return (
            FileFormat.PARQUET,
            0.6,
            f"dataset size {size / _GB:.2f} GB benefits from Parquet's compression and splittable row groups",
        )
    if size <= _SMALL_DATASET_BYTES:
        return (
            FileFormat.CSV,
            0.2,
            f"dataset size {size / _MB:.1f} MB is small enough that CSV's simplicity outweighs Parquet's overhead",
        )
    return None


def _wide_schema_rule(req: PipelineRequest) -> Vote | None:
    cols = req.source.num_columns
    if cols is None or cols < _WIDE_SCHEMA_COLUMNS:
        return None
    if {OperationType.FILTER, OperationType.AGGREGATE} & set(req.operations):
        return (
            FileFormat.PARQUET,
            0.4,
            f"{cols} columns with selective operations benefit from Parquet's column pruning",
        )
    return None


def _nested_schema_rule(req: PipelineRequest) -> Vote | None:
    schema = req.source.schema
    if not schema:
        return None
    if any(any(hint in col_type.lower() for hint in _NESTED_TYPE_HINTS) for col_type in schema.values()):
        return FileFormat.JSON, 0.5, "schema contains nested/semi-structured types better represented as JSON"
    return None


def _append_write_mode_rule(req: PipelineRequest) -> Vote | None:
    dest = req.destination
    if dest is None or not isinstance(dest.source, FileSource):
        return None
    if dest.source.write_mode == WriteMode.APPEND and EngineType.SPARK in req.available_engines:
        return (
            FileFormat.DELTA,
            0.45,
            "append writes benefit from Delta Lake's ACID transactions and schema enforcement",
        )
    return None


DEFAULT_RULES: list[Rule] = [
    _operation_rule,
    _no_operations_rule,
    _size_bytes_rule,
    _wide_schema_rule,
    _nested_schema_rule,
    _append_write_mode_rule,
]


class RuleBasedFormatProfiler(Profiler[FormatRecommendation]):
    """Recommends an output file format from dataset size, operations, schema, and write-mode heuristics."""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self._rules = rules if rules is not None else list(DEFAULT_RULES)

    @property
    def name(self) -> str:
        return "rule_based_format_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return True

    def profile(self, request: PipelineRequest) -> ProfilingResult[FormatRecommendation]:
        scored = aggregate_votes(request, self._rules, set(FileFormat))

        recommendations = [
            FormatRecommendation(format=fmt, confidence=confidence, reasoning=reasoning)
            for fmt, confidence, reasoning in scored
        ]

        return ProfilingResult(request=request, recommendations=recommendations)
