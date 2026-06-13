from dataclasses import dataclass
from typing import Callable

from core.profiler import Profiler
from models.models import (
    EngineRecommendation,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
    ProfilingResult,
)

Vote = tuple[EngineType, float, str]
Rule = Callable[[PipelineRequest], Vote | None]

_GB = 1024 ** 3
_LARGE_DATASET_BYTES = 10 * _GB
_MEDIUM_DATASET_BYTES = 1 * _GB
_LARGE_ROW_COUNT = 100_000_000


def _required_engine_rule(req: PipelineRequest) -> Vote | None:
    if req.required_engine is not None:
        return req.required_engine, 1.0, "explicitly required by caller"
    return None


def _size_bytes_rule(req: PipelineRequest) -> Vote | None:
    size = req.source.size_bytes
    if size is None:
        return None
    if size >= _LARGE_DATASET_BYTES:
        return EngineType.SPARK, 0.8, f"dataset size {size / _GB:.1f} GB exceeds {_LARGE_DATASET_BYTES // _GB} GB threshold"
    if size <= _MEDIUM_DATASET_BYTES:
        return EngineType.DUCKDB, 0.6, f"dataset size {size / _GB:.2f} GB fits comfortably in DuckDB"
    return None


def _datafusion_size_rule(req: PipelineRequest) -> Vote | None:
    size = req.source.size_bytes
    if size is None:
        return None
    if size <= _MEDIUM_DATASET_BYTES:
        return EngineType.DATAFUSION, 0.55, f"dataset size {size / _GB:.2f} GB suits DataFusion's single-node Arrow execution"
    return None


def _row_count_rule(req: PipelineRequest) -> Vote | None:
    rows = req.source.row_count
    if rows is None:
        return None
    if rows >= _LARGE_ROW_COUNT:
        return EngineType.SPARK, 0.5, f"{rows:,} rows benefits from distributed processing"
    return EngineType.DUCKDB, 0.4, f"{rows:,} rows is well within single-node capacity"


def _datafusion_row_count_rule(req: PipelineRequest) -> Vote | None:
    rows = req.source.row_count
    if rows is None:
        return None
    if rows < _LARGE_ROW_COUNT:
        return EngineType.DATAFUSION, 0.35, f"{rows:,} rows is well within DataFusion's single-node capacity"
    return None


def _format_rule(req: PipelineRequest) -> Vote | None:
    src = req.source.source
    if not isinstance(src, FileSource):
        return None
    fmt = src.format
    if fmt in (FileFormat.ORC, FileFormat.DELTA):
        return EngineType.SPARK, 0.4, f"{fmt.value} format is native to the Spark/Hadoop ecosystem"
    if fmt == FileFormat.PARQUET:
        return EngineType.DATAFUSION, 0.35, f"Parquet is Arrow-native; DataFusion reads it without conversion overhead"
    if fmt in (FileFormat.CSV, FileFormat.JSON):
        return EngineType.DUCKDB, 0.3, f"{fmt.value} is natively supported by DuckDB"
    return None


def _operation_rule(req: PipelineRequest) -> Vote | None:
    ops = set(getattr(req, "operations", []))
    if not ops:
        return None
    heavy_ops = {OperationType.JOIN, OperationType.WINDOW}
    if heavy_ops & ops and req.source.size_bytes and req.source.size_bytes >= _MEDIUM_DATASET_BYTES:
        return EngineType.SPARK, 0.4, f"heavy operations {[o.value for o in heavy_ops & ops]} on a large dataset favour Spark"
    return None


_RULES: list[Rule] = [
    _required_engine_rule,
    _size_bytes_rule,
    _datafusion_size_rule,
    _row_count_rule,
    _datafusion_row_count_rule,
    _format_rule,
    _operation_rule,
]


@dataclass
class _Tally:
    total_weight: float = 0.0
    reasons: list[str] = None

    def __post_init__(self):
        self.reasons = []

    def add(self, weight: float, reason: str) -> None:
        self.total_weight += weight
        self.reasons.append(reason)


class RuleBasedEngineProfiler(Profiler):

    @property
    def name(self) -> str:
        return "rule_based_engine_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return True

    def profile(self, request: PipelineRequest) -> ProfilingResult:
        available = set(request.available_engines)
        tallies: dict[EngineType, _Tally] = {e: _Tally() for e in available}

        for rule in _RULES:
            vote = rule(request)
            if vote is not None:
                engine, weight, reason = vote
                if engine in available:
                    tallies[engine].add(weight, reason)

        total = sum(t.total_weight for t in tallies.values()) or 1.0

        recommendations = [
            EngineRecommendation(
                engine=engine,
                confidence=round(tally.total_weight / total, 3),
                reasoning="; ".join(tally.reasons) or "no applicable rules matched",
            )
            for engine, tally in tallies.items()
            if tally.total_weight > 0
        ]
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return ProfilingResult(request=request, recommendations=recommendations)
