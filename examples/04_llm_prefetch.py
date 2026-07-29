import csv
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.models import (
    DatasetInfo,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
)
from profilers.engine.llm_engine_profiler import LLMEngineProfiler
from profilers.prefetching.prefetching_profiler import PrefetchingProfiler

# ---------------------------------------------------------------------------
# Step 1 - Create a synthetic CSV to simulate a real source file
# ---------------------------------------------------------------------------

def _make_synthetic_csv(path: str, n_rows: int, n_cols: int) -> None:
    headers = [f"col_{i}" for i in range(n_cols)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(n_rows):
            writer.writerow([i * j for j in range(n_cols)])


with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
    tmp_path = tmp.name

_make_synthetic_csv(tmp_path, n_rows=2_000_000, n_cols=25)
print(f"Synthetic CSV written to: {tmp_path}")
print(f"  On-disk size: {os.path.getsize(tmp_path) / (1024**2):.1f} MB\n")


# ---------------------------------------------------------------------------
# Step 2 - Define the prefetch function (stdlib only, no engine dependency)
# ---------------------------------------------------------------------------

def file_prefetch(request: PipelineRequest) -> DatasetInfo:
    """Read size from disk and row/column count from the CSV header + line count."""
    src = request.source.source
    assert isinstance(src, FileSource), "file_prefetch only handles file sources"

    size_bytes = os.path.getsize(src.path)

    with open(src.path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        num_columns = len(headers)
        row_count = sum(1 for _ in reader)

    schema = {h: "unknown" for h in headers}

    print("Prefetch complete:")
    print(f"  size_bytes  = {size_bytes:,}")
    print(f"  row_count   = {row_count:,}")
    print(f"  num_columns = {num_columns}")
    print()

    return DatasetInfo(
        source=src,
        size_bytes=size_bytes,
        row_count=row_count,
        num_columns=num_columns,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Step 3 - Build the profiler chain: prefetch -> Claude
# ---------------------------------------------------------------------------

llm_profiler = LLMEngineProfiler()  # reads ANTHROPIC_API_KEY from env

profiler = PrefetchingProfiler(
    prefetch_fn=file_prefetch,
    delegate=llm_profiler,
)

# ---------------------------------------------------------------------------
# Step 4 - Profile a request that has no metadata (just a path + format)
# ---------------------------------------------------------------------------

request = PipelineRequest(
    source=DatasetInfo(
        source=FileSource(path=tmp_path, format=FileFormat.CSV),
        # size_bytes, row_count, num_columns intentionally omitted
    ),
    operations=[OperationType.FILTER, OperationType.AGGREGATE, OperationType.JOIN],
    available_engines=list(EngineType),
)

print("Profiling request (no metadata provided upfront) ...")
result = profiler.profile(request)

print("Recommendations:")
print("-" * 40)
for rec in result.recommendations:
    marker = " <-- recommended" if rec == result.best else ""
    print(f"  {rec.engine.value:<12} confidence={rec.confidence:.3f}{marker}")
    print(f"               {rec.reasoning}")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_ENGINE_COLORS = {"duckdb": "#f5a623", "polars": "#7b4fd6", "spark": "#e84040"}

engines = [rec.engine.value for rec in result.recommendations]
confs   = [rec.confidence   for rec in result.recommendations]
colors  = [_ENGINE_COLORS.get(e, "#aaaaaa") for e in engines]

fig, ax = plt.subplots(figsize=(7, 3))
bars = ax.barh(engines, confs, color=colors, edgecolor="white", height=0.5)
ax.set_xlim(0, 1.1)
ax.set_xlabel("Confidence", fontsize=11)
ax.set_title("Prefetch + LLM profiler — engine recommendation", fontsize=11, fontweight="bold")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
for bar, conf in zip(bars, confs):
    ax.text(conf + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{conf:.3f}", va="center", fontsize=10)
ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
out = Path(__file__).parent / "04_llm_prefetch_result.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved -> {out}")
plt.show()

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

os.unlink(tmp_path)
