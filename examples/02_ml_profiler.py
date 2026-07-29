import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import random

import matplotlib.pyplot as plt
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
from profilers.common.features import FEATURE_NAMES, extract
from profilers.engine.ml_engine_profiler import MLEngineProfiler
from profilers.engine.rule_based_engine_profiler import RuleBasedEngineProfiler

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

_ENGINE_COLORS = {"duckdb": "#f5a623", "polars": "#7b4fd6", "spark": "#e84040"}
_ALL_ENGINES   = [e.value for e in EngineType]

rule_profiler = RuleBasedEngineProfiler()
_compare_results: list[tuple[str, object, object]] = []


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
    _compare_results.append((label, rule_result, ml_result))


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

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

n = len(_compare_results)
ncols = 2
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3))
fig.suptitle("Rule-based vs ML profiler — confidence per engine", fontsize=13, fontweight="bold")
axes = axes.flatten()

width = 0.35

for i, (label, rule_result, ml_result) in enumerate(_compare_results):
    ax = axes[i]
    engines = _ALL_ENGINES
    rule_scores = {rec.engine.value: rec.confidence for rec in rule_result.recommendations}
    ml_scores   = {rec.engine.value: rec.confidence for rec in ml_result.recommendations}

    x = np.arange(len(engines))
    rule_vals = [rule_scores.get(e, 0.0) for e in engines]
    ml_vals   = [ml_scores.get(e, 0.0) for e in engines]
    colors    = [_ENGINE_COLORS.get(e, "#aaaaaa") for e in engines]

    ax.bar(x - width / 2, rule_vals, width, label="rule-based", color=colors, alpha=0.9, edgecolor="white")
    ax.bar(x + width / 2, ml_vals,   width, label="ML",         color=colors, alpha=0.45, edgecolor="white", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(engines, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Confidence", fontsize=9)
    ax.set_title(label[:52] + ("…" if len(label) > 52 else ""), fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    rule_best = rule_result.best
    ml_best   = ml_result.best
    if rule_best and ml_best and rule_best.engine != ml_best.engine:
        ax.set_facecolor("#fff4f4")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
out = Path(__file__).parent / "02_ml_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved -> {out}")
plt.show()
