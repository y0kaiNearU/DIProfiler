import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt

from models.models import (
    DatasetInfo,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
)
from profilers.rule_based_engine_profiler import DEFAULT_RULES, RuleBasedEngineProfiler

_MB = 1024 ** 2
_GB = 1024 ** 3

_ENGINE_COLORS = {"duckdb": "#f5a623", "datafusion": "#4a90d9", "polars": "#7b4fd6", "dask": "#38a169", "spark": "#e84040"}
_ALL_ENGINES   = [e.value for e in EngineType if e != EngineType.SPARK] + ["spark"]

profiler = RuleBasedEngineProfiler()
_scenario_results: list[tuple[str, dict[str, float]]] = []


def show(label: str, request: PipelineRequest, p: RuleBasedEngineProfiler | None = None) -> None:
    result = (p or profiler).profile(request)
    print(f"\n{label}")
    print("-" * len(label))
    if not result.recommendations:
        print("  (no recommendations)")
        _scenario_results.append((label, {}))
        return
    for rec in result.recommendations:
        marker = " <-- best" if rec == result.best else ""
        print(f"  {rec.engine.value:<12} confidence={rec.confidence:.3f}  {rec.reasoning}{marker}")
    _scenario_results.append((label, {rec.engine.value: rec.confidence for rec in result.recommendations}))


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

# ---------------------------------------------------------------------------
# Custom rule: wide-table boost for DataFusion
#
# DataFusion's columnar Arrow execution is especially efficient when a query
# touches many columns (projections over wide tables). This rule doesn't exist
# in the default set — we add it on top to encode that domain knowledge.
# ---------------------------------------------------------------------------

def _wide_table_rule(req: PipelineRequest):
    cols = req.source.num_columns
    if cols is not None and cols > 50:
        return (
            EngineType.DATAFUSION,
            0.45,
            f"{cols} columns benefits from DataFusion's columnar Arrow execution",
        )
    return None


custom_profiler = RuleBasedEngineProfiler(rules=DEFAULT_RULES + [_wide_table_rule])

print("\n--- Custom rule: wide-table boost ---")
show(
    "Wide Parquet (2 GB, 120 cols) -custom rule fires",
    make_request(2 * _GB, FileFormat.PARQUET, ops=[OperationType.FILTER]),
    p=custom_profiler,
)
show(
    "Narrow CSV (500 MB, 8 cols) -custom rule silent",
    make_request(500 * _MB, FileFormat.CSV),
    p=custom_profiler,
)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

n = len(_scenario_results)
ncols = 2
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 2.4))
fig.suptitle("Rule-based profiler — confidence scores by scenario", fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, (label, scores) in enumerate(_scenario_results):
    ax = axes[i]
    engines = [e for e in _ALL_ENGINES if e in scores]
    confs   = [scores[e] for e in engines]
    colors  = [_ENGINE_COLORS.get(e, "#aaaaaa") for e in engines]
    bars = ax.barh(engines, confs, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Confidence", fontsize=9)
    ax.set_title(label[:50] + ("…" if len(label) > 50 else ""), fontsize=9, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    for bar, conf in zip(bars, confs):
        ax.text(conf + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{conf:.2f}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
out = Path(__file__).parent / "01_rule_based_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved -> {out}")
plt.show()
