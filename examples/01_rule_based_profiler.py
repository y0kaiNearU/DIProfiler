import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.models import (
    DatasetInfo,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
)
from profilers.engine_profiler import RuleBasedEngineProfiler

_MB = 1024 ** 2
_GB = 1024 ** 3

profiler = RuleBasedEngineProfiler()


def show(label: str, request: PipelineRequest) -> None:
    result = profiler.profile(request)
    print(f"\n{label}")
    print("-" * len(label))
    if not result.recommendations:
        print("  (no recommendations)")
        return
    for rec in result.recommendations:
        marker = " <-- best" if rec == result.best else ""
        print(f"  {rec.engine.value:<12} confidence={rec.confidence:.3f}  {rec.reasoning}{marker}")


def make_request(size_bytes: int, fmt: FileFormat, ops: list[OperationType] | None = None) -> PipelineRequest:
    rows = size_bytes // 200
    return PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/input", format=fmt),
            size_bytes=size_bytes,
            row_count=rows,
            num_columns=20,
        ),
        operations=ops or [],
        available_engines=list(EngineType),
    )


# --- Scenarios ---

show(
    "Small CSV (500 MB) -single-node territory",
    make_request(500 * _MB, FileFormat.CSV),
)

show(
    "Large ORC (50 GB) -Spark/Hadoop ecosystem",
    make_request(50 * _GB, FileFormat.ORC),
)

show(
    "Medium Parquet (800 MB) -Arrow-native",
    make_request(800 * _MB, FileFormat.PARQUET),
)

show(
    "Large Parquet (20 GB) + window + join -heavy ops on big data",
    make_request(20 * _GB, FileFormat.PARQUET, ops=[OperationType.WINDOW, OperationType.JOIN]),
)

show(
    "Small JSON (50 MB) -DuckDB home turf",
    make_request(50 * _MB, FileFormat.JSON),
)

show(
    "Small CSV (100 MB) -only Spark available (e.g. cluster-only environment)",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/input", format=FileFormat.CSV),
            size_bytes=100 * _MB,
            row_count=500_000,
            num_columns=10,
        ),
        available_engines=[EngineType.SPARK],
    ),
)
