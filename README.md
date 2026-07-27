# DIProfiler

Profiles data pipeline requests and recommends the best query engine (pandas, DuckDB, Polars, Dask, Spark, Arrow) based on dataset characteristics. Includes rule-based, ML-based, prefetching, and LLM-based profilers.

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Running the examples

```bash
uv run python examples/01_rule_based_profiler.py
uv run python examples/02_ml_profiler.py
uv run python examples/03_diprofiler.py
uv run python examples/04_llm_prefetch.py
uv run python examples/05_benchmark.py
```

## Benchmark (`05_benchmark.py`)

Generates CSVs of increasing size, times each available engine on the same aggregation query, and plots execution time with each profiler's recommendation marked on the chart.

### Optional engines

Install only the engines you want to benchmark:

```bash
# DuckDB
uv add duckdb

# Polars
uv add polars

# Dask
uv add "dask[dataframe,distributed]"

# Spark — requires Java 8+ installed separately
uv pip install "pyspark==3.5.5"
```

### Optional LLM profiler

```bash
uv add anthropic
```

Set your API key before running:

```bash
$env:ANTHROPIC_API_KEY = "sk-..."   # PowerShell
# or
export ANTHROPIC_API_KEY="sk-..."   # bash
```

### Install everything at once

```bash
uv add duckdb polars "dask[dataframe,distributed]" anthropic
uv pip install "pyspark==3.5.5"   # optional, requires Java 8+
```

Any engine or profiler whose dependency is missing is skipped automatically — the benchmark always runs with at least pandas and the rule-based, ML, and prefetching profilers.

Arrow is always available (core dependency) for engine recommendation via `DIProfiler`/`RuleBasedEngineProfiler`, but isn't wired into this benchmark script's timing runners yet.

## Project structure

```
src/
  core/        base interfaces
  engines/     duckdb / polars / dask / spark / pandas / arrow adapters
  models/      shared data models
  profilers/   rule-based, ML, prefetching, LLM profilers
examples/      runnable scripts (01–05)
tests/         unit tests
```

## Running tests

```bash
uv run pytest
```
