import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.models import (
    DatasetInfo,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
)
from profilers.engine_profiler import RuleBasedEngineProfiler
from profilers.features import FEATURE_NAMES, extract
from profilers.ml_engine_profiler import MLEngineProfiler

_MB = 1024 ** 2
_GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Step 1 — Generate training data (your code; swap in real telemetry here)
# ---------------------------------------------------------------------------

def _random_request(rng: random.Random) -> PipelineRequest:
    size = rng.randint(_MB, 100 * _GB)
    rows = max(1, size // rng.randint(50, 5_000))
    return PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="synthetic", format=rng.choice(list(FileFormat))),
            size_bytes=size,
            row_count=rows,
            num_columns=rng.randint(1, 200),
        ),
        operations=rng.sample(list(OperationType), k=rng.randint(0, len(OperationType))),
        available_engines=list(EngineType),
    )


def build_training_data(n: int = 2_000, seed: int = 42):
    label_profiler = RuleBasedEngineProfiler()
    rng = random.Random(seed)
    X, y = [], []
    while len(X) < n:
        req = _random_request(rng)
        result = label_profiler.profile(req)
        if result.best:
            X.append(extract(req))
            y.append(result.best.engine.value)
    return np.array(X, dtype=np.float32), np.array(y)


print("Generating synthetic training samples ...")
X, y = build_training_data(n=2_000)
print(f"  {len(X)} samples  |  features: {len(FEATURE_NAMES)}")
print(f"  label distribution: { {v: (y == v).sum() for v in np.unique(y)} }")

# ---------------------------------------------------------------------------
# Step 2 — Train any sklearn classifier and wrap it
# ---------------------------------------------------------------------------

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X, y)

ml_profiler = MLEngineProfiler(model)

# ---------------------------------------------------------------------------
# Step 3 — Compare rule-based vs ML on the same requests
# ---------------------------------------------------------------------------

rule_profiler = RuleBasedEngineProfiler()


def compare(label: str, request: PipelineRequest) -> None:
    rule_result = rule_profiler.profile(request)
    ml_result = ml_profiler.profile(request)

    rule_best = rule_result.best
    ml_best = ml_result.best

    print(f"\n{label}")
    print("-" * len(label))
    print(f"  rule-based -> {rule_best.engine.value:<12} (confidence {rule_best.confidence:.3f})" if rule_best else "  rule-based -> (no recommendation)")
    print(f"  ml         -> {ml_best.engine.value:<12} (confidence {ml_best.confidence:.3f})" if ml_best else "  ml         -> (no recommendation)")

    if rule_best and ml_best and rule_best.engine != ml_best.engine:
        print("  ** DISAGREE **")


compare("Small CSV (200 MB)", PipelineRequest(
    source=DatasetInfo(
        source=FileSource(path="x", format=FileFormat.CSV),
        size_bytes=200 * _MB, row_count=1_000_000, num_columns=15,
    ),
    available_engines=list(EngineType),
))

compare("Large ORC (40 GB)", PipelineRequest(
    source=DatasetInfo(
        source=FileSource(path="x", format=FileFormat.ORC),
        size_bytes=40 * _GB, row_count=500_000_000, num_columns=80,
    ),
    available_engines=list(EngineType),
))

compare("Medium Parquet (900 MB) + aggregation", PipelineRequest(
    source=DatasetInfo(
        source=FileSource(path="x", format=FileFormat.PARQUET),
        size_bytes=900 * _MB, row_count=5_000_000, num_columns=30,
    ),
    operations=[OperationType.AGGREGATE],
    available_engines=list(EngineType),
))

compare("Large Parquet (15 GB) + window + join", PipelineRequest(
    source=DatasetInfo(
        source=FileSource(path="x", format=FileFormat.PARQUET),
        size_bytes=15 * _GB, row_count=200_000_000, num_columns=50,
    ),
    operations=[OperationType.WINDOW, OperationType.JOIN],
    available_engines=list(EngineType),
))
