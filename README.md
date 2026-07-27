# DIProfiler

Profiles data pipeline requests and recommends the best query engine (pandas, DuckDB, Polars, Dask, Spark, Arrow, Modin) based on dataset characteristics. Includes rule-based, ML-based, prefetching, and LLM-based profilers.

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

The core library depends on nothing but `narwhals` and `packaging` — every
engine and ML/LLM feature is an optional extra, installed only if you want
it, the same way `narwhals` itself stays dependency-free until you pick a
backend:

```bash
uv sync
```

That alone gets you `DIProfiler`/`RuleBasedEngineProfiler`-style profiling
and recommendations with zero engine libraries installed. Loading or writing
data needs the engine's own package; install only the extras you use:

```bash
uv sync --extra polars      # or duckdb, dask, spark, arrow, modin
uv sync --extra ml          # numpy + scikit-learn + joblib, for MLEngineProfiler
uv sync --extra llm         # anthropic, for LLMEngineProfiler/LLMPartitionProfiler
uv sync --extra openai      # openai, same profilers via a different client
uv sync --extra all         # everything above
```

`uv sync` also installs the `dev` dependency group by default (pytest, every
engine, numpy/scikit-learn — everything the test suite in this repo
exercises), so the extras above matter mainly if you want a leaner
environment than "run the full test suite" — pass `--no-default-groups`
alongside `--extra ...` for that. A missing engine's loader/writer raises a
friendly `ImportError` telling you what to add (e.g. `Polars is required: uv
add polars`).

## Running the examples

See [examples/README.md](examples/README.md) — they need their own
dependency group (`matplotlib`, demo ML training, etc.) separate from both
the lean core and the test suite's `dev` group.

## Project structure

```
src/
  core/        base interfaces
  engines/     duckdb / polars / dask / spark / pandas / arrow / modin adapters
  models/      shared data models
  profilers/   rule-based, ML, prefetching, LLM profilers
examples/      runnable scripts (01–05), own dependency group — see examples/README.md
tests/         unit tests
```

## Running tests

`uv sync`'s default `dev` group already installs every engine and ML
dependency the test suite exercises (pandas, DuckDB, Polars, Dask, Spark,
Arrow, Modin, numpy, scikit-learn) — no extra flags needed:

```bash
uv run pytest
```
