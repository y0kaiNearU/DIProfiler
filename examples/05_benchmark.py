"""
Example 5 - Benchmark: do profiler recommendations match actual performance?

Generates CSV files of increasing size, times each available engine on the same
aggregation query (N_REPEATS runs each, median kept), then plots execution time
with each profiler's recommendation marked on the chart.

Engines tested:
  - pandas       always (baseline)
  - duckdb       if installed  (uv add duckdb)
  - datafusion   if installed  (uv add datafusion pyarrow)
  - spark        if installed manually — NOT a project dependency, install separately:
                   uv pip install "pyspark==3.5.5"   # requires Java 8+
                 Runs in local[*] mode (single machine, not a cluster).

Profilers compared:
  - RuleBasedEngineProfiler   always
  - MLEngineProfiler          always (bootstrapped from rule-based labels)
  - PrefetchingProfiler       always (reads real row_count from file, delegates to rule-based)
  - LLMEngineProfiler         if anthropic is installed  (uv add anthropic)
"""
import csv as _csv_module
import importlib.util
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
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
from profilers.features import extract
from profilers.ml_engine_profiler import MLEngineProfiler
from profilers.prefetching_profiler import PrefetchingProfiler

logging.getLogger("pyspark").setLevel(logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)

_MB = 1024 ** 2
_GB = 1024 ** 3

_available_engine_types: list[EngineType] = []

if importlib.util.find_spec("duckdb"):
    import duckdb
    _available_engine_types.append(EngineType.DUCKDB)

if importlib.util.find_spec("datafusion") and importlib.util.find_spec("pyarrow"):
    import datafusion
    _available_engine_types.append(EngineType.DATAFUSION)

_spark = None
if importlib.util.find_spec("pyspark"):
    try:
        from pyspark.sql import SparkSession
        _spark = (
            SparkSession.builder
            .master("local[*]")
            .appName("DIProfiler-benchmark")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel("ERROR")
        _available_engine_types.append(EngineType.SPARK)
        print("Spark session started (local[*] mode)")
    except Exception as e:
        print(f"Spark init failed: {e}")

print(f"Library engines available: {[e.value for e in _available_engine_types]}")
print("Baseline engine:           pandas\n")

# 1. Rule-based
_rule_profiler = RuleBasedEngineProfiler()

# 2. ML — bootstrap 1 000 samples from rule-based labels, train a small RF
print("Training ML profiler ...")
_rng = random.Random(42)
_X, _y = [], []
while len(_X) < 1_000:
    _size = _rng.randint(_MB, 100 * _GB)
    _tmp_req = PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="s", format=_rng.choice(list(FileFormat))),
            size_bytes=_size,
            row_count=max(1, _size // 500),
            num_columns=_rng.randint(1, 100),
        ),
        operations=_rng.sample(list(OperationType), k=_rng.randint(0, 3)),
        available_engines=list(EngineType),
    )
    _res = _rule_profiler.profile(_tmp_req)
    if _res.best:
        _X.append(extract(_tmp_req))
        _y.append(_res.best.engine.value)
_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
_clf.fit(np.array(_X, dtype=np.float32), np.array(_y))
_ml_profiler = MLEngineProfiler(_clf)
print("Done.\n")

# 3. Prefetching — reads actual row_count from the CSV, then delegates to rule-based.
#    Results are cached so each file is read only once across all profiler calls.
_prefetch_cache: dict[str, DatasetInfo] = {}

def _file_prefetch(request: PipelineRequest) -> DatasetInfo:
    src = request.source.source
    path = src.path
    if path not in _prefetch_cache:
        with open(path, newline="") as f:
            reader = _csv_module.reader(f)
            headers = next(reader)
            row_count = sum(1 for _ in reader)
        _prefetch_cache[path] = DatasetInfo(
            source=src,
            size_bytes=request.source.size_bytes,
            row_count=row_count,
            num_columns=len(headers),
        )
    return _prefetch_cache[path]

_prefetch_profiler = PrefetchingProfiler(
    prefetch_fn=_file_prefetch,
    delegate=RuleBasedEngineProfiler(),
)

# 4. LLM — optional, requires anthropic package
_llm_profiler = None
if importlib.util.find_spec("anthropic"):
    from profilers.llm_engine_profiler import LLMEngineProfiler
    _llm_profiler = LLMEngineProfiler()
    print("LLM profiler enabled (ANTHROPIC_API_KEY must be set)\n")
else:
    print("LLM profiler skipped (uv add anthropic to enable)\n")

_PROFILERS: dict[str, object] = {
    "rule_based":  _rule_profiler,
    "ml":          _ml_profiler,
    "prefetching": _prefetch_profiler,
}
if _llm_profiler:
    _PROFILERS["llm"] = _llm_profiler

# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------

SIZES_MB = [5, 20, 50, 100, 200, 400]
N_REPEATS = 3
N_CATEGORIES = 50


def _make_csv(path: str, target_bytes: int) -> int:
    n_rows = max(5_000, target_bytes // 72)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "category": rng.integers(0, N_CATEGORIES, n_rows).astype(str),
        "value_a":  rng.uniform(0, 1_000, n_rows),
        "value_b":  rng.uniform(0, 1_000, n_rows),
        "value_c":  rng.uniform(0, 1_000, n_rows),
        "value_d":  rng.uniform(0, 1_000, n_rows),
    })
    df.to_csv(path, index=False)
    return os.path.getsize(path)


_SQL = (
    "SELECT category, COUNT(*) AS n, SUM(value_a) AS total "
    "FROM {src} GROUP BY category ORDER BY category"
)


def _run_pandas(path: str) -> float:
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    df.groupby("category").agg(n=("value_a", "count"), total=("value_a", "sum"))
    return time.perf_counter() - t0


def _run_duckdb(path: str) -> float:
    t0 = time.perf_counter()
    con = duckdb.connect()
    con.execute(_SQL.format(src=f"read_csv_auto('{path}')")).fetchall()
    con.close()
    return time.perf_counter() - t0


def _run_datafusion(path: str) -> float:
    t0 = time.perf_counter()
    ctx = datafusion.SessionContext()
    ctx.register_csv("data", path)
    ctx.sql(_SQL.format(src="data")).collect()
    return time.perf_counter() - t0


def _run_spark(path: str) -> float:
    t0 = time.perf_counter()
    df = _spark.read.csv(path, header=True, inferSchema=True)
    df.createOrReplaceTempView("data")
    _spark.sql(_SQL.format(src="data")).collect()
    return time.perf_counter() - t0


_RUNNERS: dict[str, object] = {"pandas": _run_pandas}
if EngineType.DUCKDB in _available_engine_types:
    _RUNNERS["duckdb"] = _run_duckdb
if EngineType.DATAFUSION in _available_engine_types:
    _RUNNERS["datafusion"] = _run_datafusion
if EngineType.SPARK in _available_engine_types:
    _RUNNERS["spark"] = _run_spark


def _median_time(runner, path: str) -> float:
    times = sorted(runner(path) for _ in range(N_REPEATS))
    return times[N_REPEATS // 2]


if _spark is not None:
    print("Warming up Spark JVM ...")
    with tempfile.TemporaryDirectory() as _tmp:
        _warmup = str(Path(_tmp) / "warmup.csv")
        _make_csv(_warmup, 1 * _MB)
        _run_spark(_warmup)
    print("Spark warm-up done.\n")


# engine timing: engine_name -> [(actual_mb, median_seconds), ...]
engine_results: dict[str, list[tuple[float, float]]] = {name: [] for name in _RUNNERS}

# profiler recommendations: profiler_name -> {actual_mb -> engine_value | None}
profiler_recs: dict[str, dict[float, str | None]] = {name: {} for name in _PROFILERS}

with tempfile.TemporaryDirectory() as tmpdir:
    for target_mb in SIZES_MB:
        csv_path = str(Path(tmpdir) / f"bench_{target_mb}mb.csv")
        actual_bytes = _make_csv(csv_path, target_mb * _MB)
        actual_mb = round(actual_bytes / _MB, 1)

        req = PipelineRequest(
            source=DatasetInfo(
                source=FileSource(path=csv_path, format=FileFormat.CSV),
                size_bytes=actual_bytes,
                num_columns=5,
            ),
            available_engines=_available_engine_types or list(EngineType),
        )

        # Collect recommendations from all profilers
        rec_summary = {}
        for p_name, p in _PROFILERS.items():
            if p.can_handle(req):
                rec = p.profile(req)
                engine_val = rec.best.engine.value if rec.best else None
            else:
                engine_val = None
            profiler_recs[p_name][actual_mb] = engine_val
            rec_summary[p_name] = engine_val or "(none)"

        print(f"{actual_mb:>6.0f} MB")
        for p_name, engine_val in rec_summary.items():
            print(f"          profiler [{p_name:<12}] -> {engine_val}")

        # Time each engine
        for name, runner in _RUNNERS.items():
            t = _median_time(runner, csv_path)
            engine_results[name].append((actual_mb, t))
            print(f"          engine   [{name:<12}]    {t:.3f}s")
        print()

if _spark is not None:
    _spark.stop()

_ENGINE_COLORS  = {"pandas": "#aaaaaa", "duckdb": "#f5a623", "datafusion": "#4a90d9", "spark": "#e84040"}
_ENGINE_MARKERS = {"pandas": "^",       "duckdb": "o",       "datafusion": "s",       "spark": "D"}
_ENGINE_LABELS  = {"pandas": "pandas (baseline)", "duckdb": "DuckDB", "datafusion": "DataFusion", "spark": "Spark (local[*])"}

# Marker shape per profiler recommendation
_PROFILER_STYLE = {
    "rule_based":  ("*", 380, "rule-based rec."),
    "ml":          ("D", 200, "ML rec."),
    "prefetching": ("P", 200, "prefetching rec."),
    "llm":         ("s", 200, "LLM rec."),
}

fig, ax = plt.subplots(figsize=(10, 5))

for name, points in engine_results.items():
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(
        xs, ys,
        marker=_ENGINE_MARKERS.get(name, "o"),
        color=_ENGINE_COLORS.get(name, "black"),
        label=_ENGINE_LABELS.get(name, name),
        linewidth=2,
        markersize=7,
    )

_lookup: dict[str, dict[float, float]] = {
    name: {p[0]: p[1] for p in pts} for name, pts in engine_results.items()
}

# Draw one marker per profiler per size, offset horizontally to avoid stacking
_p_names = list(_PROFILERS.keys())
_n = len(_p_names)
_x_offsets = np.linspace(-0.02, 0.02, _n)  # fractional offset applied per-point

for i, p_name in enumerate(_p_names):
    if p_name not in _PROFILER_STYLE:
        continue
    marker, size, label = _PROFILER_STYLE[p_name]
    xs, ys = [], []
    for mb, rec_engine in profiler_recs[p_name].items():
        if rec_engine and rec_engine in _lookup and mb in _lookup[rec_engine]:
            xs.append(mb * (1 + _x_offsets[i]))
            ys.append(_lookup[rec_engine][mb])
    if xs:
        ax.scatter(
            xs, ys,
            marker=marker, s=size, zorder=6,
            color="black",
            edgecolors="white", linewidths=0.5,
            label=label,
        )

ax.set_xlabel("CSV file size (MB)", fontsize=12)
ax.set_ylabel("Execution time (s)", fontsize=12)
note = " — Spark in local[*] mode" if _spark else ""
ax.set_title(
    f"Engine performance vs. profiler recommendations\n"
    f"(GROUP BY + SUM, median of {N_REPEATS} runs{note})",
    fontsize=12,
)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

out_path = Path(__file__).parent / "benchmark_results.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Chart saved -> {out_path}")
plt.show()
