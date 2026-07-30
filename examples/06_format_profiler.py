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
    WriteMode,
)
from profilers.format.rule_based_format_profiler import RuleBasedFormatProfiler

_MB = 1024 ** 2
_GB = 1024 ** 3

_FORMAT_COLORS = {"csv": "#aaaaaa", "parquet": "#7b4fd6", "json": "#f5a623", "delta": "#e84040"}

profiler = RuleBasedFormatProfiler()
_scenario_results: list[tuple[str, dict[str, float]]] = []


def show(label: str, request: PipelineRequest) -> None:
    result = profiler.profile(request)
    print(f"\n{label}")
    print("-" * len(label))
    for rec in result.recommendations:
        marker = " <-- best" if rec == result.best else ""
        print(f"  {rec.format.value:<8} confidence={rec.confidence:.3f}  {rec.reasoning}{marker}")
    _scenario_results.append((label, {rec.format.value: rec.confidence for rec in result.recommendations}))


# --- Scenarios ---

show(
    "Small CSV read, no transforms -simplicity wins",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/input.csv", format=FileFormat.CSV),
            size_bytes=5 * _MB,
        ),
    ),
)

show(
    "Large dataset + filter/aggregate -columnar pruning",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/events", format=FileFormat.CSV),
            size_bytes=5 * _GB,
            num_columns=12,
        ),
        operations=[OperationType.FILTER, OperationType.AGGREGATE],
    ),
)

show(
    "Wide table (60 cols) + filter/aggregate -column pruning",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/wide_table", format=FileFormat.CSV),
            size_bytes=200 * _MB,
            num_columns=60,
        ),
        operations=[OperationType.FILTER, OperationType.AGGREGATE],
    ),
)

show(
    "Nested schema (struct/array columns) -semi-structured",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/events.json", format=FileFormat.JSON),
            size_bytes=100 * _MB,
            schema={"user_id": "int", "event": "string", "payload": "struct<...>"},
        ),
    ),
)

show(
    "Append-mode write with Spark available -Delta Lake",
    PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/input.parquet", format=FileFormat.PARQUET),
            size_bytes=2 * _GB,
        ),
        destination=DatasetInfo(
            source=FileSource(path="data/warehouse", format=FileFormat.PARQUET, write_mode=WriteMode.APPEND),
        ),
        available_engines=[EngineType.SPARK, EngineType.DUCKDB],
    ),
)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

n = len(_scenario_results)
ncols = 2
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 2.4))
fig.suptitle("Rule-based format profiler — confidence scores by scenario", fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, (label, scores) in enumerate(_scenario_results):
    ax = axes[i]
    formats = list(scores)
    confs = [scores[f] for f in formats]
    colors = [_FORMAT_COLORS.get(f, "#aaaaaa") for f in formats]
    bars = ax.barh(formats, confs, color=colors, edgecolor="white", height=0.5)
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
out = Path(__file__).parent / "06_format_profiler_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved -> {out}")
plt.show()
