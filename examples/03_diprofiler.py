import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from core.registry import ProfilerRegistry
from models.models import (
    DatasetInfo,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
)
from profilers.engine_profiler import RuleBasedEngineProfiler
from profilers.features import extract
from profilers.ml_engine_profiler import MLEngineProfiler

_MB = 1024 ** 2
_GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Build a trained ML profiler (same bootstrap as example 02)
# ---------------------------------------------------------------------------

def _bootstrap_ml(n: int = 1_500, seed: int = 42) -> MLEngineProfiler:
    labeller = RuleBasedEngineProfiler()
    rng = random.Random(seed)
    X, y = [], []
    while len(X) < n:
        size = rng.randint(_MB, 100 * _GB)
        req = PipelineRequest(
            source=DatasetInfo(
                source=FileSource(path="s", format=rng.choice(list(FileFormat))),
                size_bytes=size,
                row_count=max(1, size // 500),
                num_columns=rng.randint(1, 100),
            ),
            operations=rng.sample(list(OperationType), k=rng.randint(0, 3)),
            available_engines=list(EngineType),
        )
        result = labeller.profile(req)
        if result.best:
            X.append(extract(req))
            y.append(result.best.engine.value)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(np.array(X, dtype=np.float32), np.array(y))
    return MLEngineProfiler(clf)


print("Training ML profiler ...")
ml_profiler = _bootstrap_ml()
print("Done.\n")

# ---------------------------------------------------------------------------
# Register both profilers; the registry runs them all and returns each result
# ---------------------------------------------------------------------------

registry = ProfilerRegistry()
registry.register(RuleBasedEngineProfiler())
registry.register(ml_profiler)

print(f"Registered profilers: {registry.names}\n")


def show(label: str, request: PipelineRequest) -> None:
    results = registry.run(request)

    # Merge: keep the highest-confidence recommendation per engine
    merged: dict[EngineType, object] = {}
    for result in results:
        for rec in result.recommendations:
            if rec.engine not in merged or rec.confidence > merged[rec.engine].confidence:
                merged[rec.engine] = rec

    ranked = sorted(merged.values(), key=lambda r: r.confidence, reverse=True)
    best = ranked[0] if ranked else None

    print(label)
    print("-" * len(label))
    for rec in ranked:
        marker = " <-- recommended" if rec == best else ""
        print(f"  {rec.engine.value:<12} confidence={rec.confidence:.3f}{marker}")
    print()


show(
    "Small CSV (300 MB) -single-node territory",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/sales.csv", format=FileFormat.CSV),
            size_bytes=300 * _MB,
            row_count=1_500_000,
            num_columns=12,
        ),
        operations=[OperationType.FILTER, OperationType.AGGREGATE],
        available_engines=list(EngineType),
    ),
)

show(
    "Large Parquet (25 GB) + window + join",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/events.parquet", format=FileFormat.PARQUET),
            size_bytes=25 * _GB,
            row_count=300_000_000,
            num_columns=40,
        ),
        operations=[OperationType.JOIN, OperationType.WINDOW],
        available_engines=list(EngineType),
    ),
)

show(
    "Delta table (80 GB) -Spark ecosystem format",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/warehouse", format=FileFormat.DELTA),
            size_bytes=80 * _GB,
            row_count=1_000_000_000,
            num_columns=60,
        ),
        available_engines=list(EngineType),
    ),
)

show(
    "Medium Parquet (700 MB) -only DuckDB available",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/report.parquet", format=FileFormat.PARQUET),
            size_bytes=700 * _MB,
            row_count=3_000_000,
            num_columns=25,
        ),
        available_engines=[EngineType.DUCKDB],
    ),
)
