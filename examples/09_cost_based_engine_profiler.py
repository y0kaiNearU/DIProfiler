import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt

from models.models import DatasetInfo, EngineType, FileFormat, FileSource, PipelineRequest
from profilers.engine.cost_based_engine_profiler import CostBasedEngineProfiler, EngineCostRate

_MB = 1024 ** 2
_GB = 1024 ** 3

_ENGINE_COLORS = {"duckdb": "#aaaaaa", "dask": "#38a169", "spark": "#e84040"}

# cost_rates has no auto-detected default (see the profiler's docstring) --
# these are illustrative $/hour figures, not real cloud pricing.
_ALL_ENGINE_RATES = {
    EngineType.DUCKDB: EngineCostRate(cost_per_core_hour=0.0, cost_per_gb_hour=0.0),   # local, free
    EngineType.DASK: EngineCostRate(cost_per_core_hour=0.02, cost_per_gb_hour=0.025),  # small managed cluster
    EngineType.SPARK: EngineCostRate(cost_per_core_hour=0.08, cost_per_gb_hour=0.005),  # cluster, per-core overhead
}

_scenario_results: list[tuple[str, dict[str, float]]] = []


def show(label: str, size_bytes: int, available_engines: list[EngineType], cost_rates) -> None:
    profiler = CostBasedEngineProfiler(cost_rates)
    request = PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/input.parquet", format=FileFormat.PARQUET),
            size_bytes=size_bytes,
        ),
        available_engines=available_engines,
        available_cores=16,
        available_memory_bytes=64 * _GB,
    )
    result = profiler.profile(request)

    print(f"\n{label}")
    print("-" * len(label))
    for rec in result.recommendations:
        marker = " <-- cheapest" if rec == result.best else ""
        print(f"  {rec.engine.value:<8} confidence={rec.confidence:.3f}  {rec.reasoning}{marker}")

    _scenario_results.append((label, {rec.engine.value: rec.confidence for rec in result.recommendations}))


# ---------------------------------------------------------------------------
# Scenario 1 - all engines available, including free local DuckDB
#
# Cost-based reasoning alone doesn't check whether DuckDB can actually handle
# the data volume (that's RuleBasedEngineProfiler's job) -- it just picks the
# cheapest $/hr among the engines it's given. A free local engine trivially
# wins whenever it's in the running.
# ---------------------------------------------------------------------------

show(
    "All engines available (500 MB)",
    500 * _MB,
    [EngineType.DUCKDB, EngineType.DASK, EngineType.SPARK],
    _ALL_ENGINE_RATES,
)

# ---------------------------------------------------------------------------
# Scenario 2 - cloud-only environment (no local engine), small vs large data
#
# Dask's cheaper per-core rate wins on small jobs; Spark's cheaper per-GB
# rate catches up as the resource profiler sizes a bigger memory allocation
# for larger data.
# ---------------------------------------------------------------------------

_cloud_rates = {k: v for k, v in _ALL_ENGINE_RATES.items() if k != EngineType.DUCKDB}

show(
    "Cloud-only, small dataset (300 MB)",
    300 * _MB,
    [EngineType.DASK, EngineType.SPARK],
    _cloud_rates,
)

show(
    "Cloud-only, large dataset (50 GB) -capped at 16 cores/64 GB",
    50 * _GB,
    [EngineType.DASK, EngineType.SPARK],
    _cloud_rates,
)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

n = len(_scenario_results)
fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 3.6))
fig.suptitle("Cost-based engine profiler — cheapest engine per scenario", fontsize=12, fontweight="bold")

for ax, (label, scores) in zip(axes, _scenario_results):
    engines = list(scores)
    confs = [scores[e] for e in engines]
    colors = [_ENGINE_COLORS.get(e, "#aaaaaa") for e in engines]
    bars = ax.barh(engines, confs, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Confidence (inverse cost, normalized)", fontsize=8)
    ax.set_title(label[:40] + ("…" if len(label) > 40 else ""), fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    for bar, conf in zip(bars, confs):
        ax.text(conf + 0.02, bar.get_y() + bar.get_height() / 2, f"{conf:.2f}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
out = Path(__file__).parent / "09_cost_based_engine_profiler_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved -> {out}")
plt.show()
