import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt

from models.models import DatasetInfo, FileFormat, FileSource, PipelineRequest
from profilers.resource.rule_based_resource_profiler import RuleBasedResourceProfiler

_MB = 1024 ** 2
_GB = 1024 ** 3

_AVAILABLE_CORES = 16
_AVAILABLE_MEMORY = 64 * _GB

profiler = RuleBasedResourceProfiler()
_scenario_results: list[tuple[str, int, float, float]] = []  # label, cores, memory_gb, confidence


def show(label: str, size_bytes: int | None) -> None:
    request = PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="data/input.parquet", format=FileFormat.PARQUET),
            size_bytes=size_bytes,
        ),
        available_cores=_AVAILABLE_CORES,
        available_memory_bytes=_AVAILABLE_MEMORY,
    )
    result = profiler.profile(request)
    rec = result.best
    assert rec is not None

    print(f"\n{label}")
    print("-" * len(label))
    memory_gb = rec.memory_bytes / _GB
    budget_gb = _AVAILABLE_MEMORY / _GB
    print(f"  cores={rec.cores}/{_AVAILABLE_CORES}  memory={memory_gb:.2f}/{budget_gb:.0f} GB", end="  ")
    print(f"confidence={rec.confidence:.2f}")
    print(f"  {rec.reasoning}")

    _scenario_results.append((label, rec.cores, rec.memory_bytes / _GB, rec.confidence))


# --- Scenarios (fixed budget: 16 cores / 64 GB available) ---

show("Small dataset (500 MB)", 500 * _MB)
show("Medium dataset (8 GB)", 8 * _GB)
show("Large dataset (30 GB) -capped at available memory", 30 * _GB)
show("Huge dataset (500 GB) -capped at available cores/memory", 500 * _GB)
show("Unknown size -conservative default, lower confidence", None)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

labels = [r[0] for r in _scenario_results]
cores = [r[1] for r in _scenario_results]
memory_gb = [r[2] for r in _scenario_results]
confidences = [r[3] for r in _scenario_results]

fig, (ax_cores, ax_mem) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Rule-based resource profiler — sizing against 16 cores / 64 GB budget", fontsize=12, fontweight="bold")

colors = ["#7b4fd6" if c == 1.0 else "#aaaaaa" for c in confidences]

bars = ax_cores.barh(labels, cores, color=colors, edgecolor="white", height=0.5)
ax_cores.set_xlim(0, _AVAILABLE_CORES * 1.15)
ax_cores.axvline(_AVAILABLE_CORES, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
ax_cores.set_xlabel("Cores recommended", fontsize=9)
ax_cores.set_title("Cores", fontsize=10, fontweight="bold")
for bar, v in zip(bars, cores):
    ax_cores.text(v + 0.3, bar.get_y() + bar.get_height() / 2, str(v), va="center", fontsize=8)
ax_cores.grid(True, axis="x", alpha=0.3)

bars = ax_mem.barh(labels, memory_gb, color=colors, edgecolor="white", height=0.5)
ax_mem.set_xlim(0, _AVAILABLE_MEMORY / _GB * 1.15)
ax_mem.axvline(_AVAILABLE_MEMORY / _GB, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
ax_mem.set_xlabel("Memory recommended (GB)", fontsize=9)
ax_mem.set_title("Memory", fontsize=10, fontweight="bold")
ax_mem.set_yticklabels([])
for bar, v in zip(bars, memory_gb):
    ax_mem.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", fontsize=8)
ax_mem.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
out = Path(__file__).parent / "07_resource_profiler_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved -> {out}")
plt.show()
