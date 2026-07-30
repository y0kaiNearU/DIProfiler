# Running the examples

The core library (`narwhals` + `packaging` only) doesn't include the engine
libraries, matplotlib, numpy, or scikit-learn these scripts use for plotting
and demo model training. They live in the `examples` dependency group.

`uv`'s `dev` group is installed by default on every `uv sync`/`uv run` — pass
`--no-default-groups` alongside `--group examples` if you want *just* what
the examples need, without also pulling in pytest/mypy/pyspark:

```bash
uv sync --no-default-groups --group examples
uv run --no-default-groups --group examples python examples/01_rule_based_profiler.py
```

If you're already working in this repo (i.e. you ran a plain `uv sync` and
have the `dev` group installed), you don't need any of that — the `dev`
group already installs everything `examples` does. Just run:

```bash
uv run python examples/01_rule_based_profiler.py
uv run python examples/02_ml_profiler.py
uv run python examples/03_diprofiler.py
uv run python examples/04_llm_prefetch.py
uv run python examples/05_benchmark.py
uv run python examples/06_format_profiler.py
uv run python examples/07_resource_profiler.py
uv run python examples/08_partition_profiler.py
uv run python examples/09_cost_based_engine_profiler.py
```

## What's in the `examples` group

`matplotlib`, `pandas`, `duckdb`, `polars`, `dask[dataframe,distributed]`,
`numpy`, `scikit-learn`, and `anthropic`. That covers every example except
Spark, which is deliberately left out (see below).

## Script-specific notes

- **`01_rule_based_profiler.py`** — no extra setup; pure rule-based profiling
  and a matplotlib chart.
- **`02_ml_profiler.py`** / **`03_diprofiler.py`** — bootstrap a small
  `RandomForestClassifier` from `scikit-learn` to demo the ML profiler.
- **`04_llm_prefetch.py`** — needs a real Claude API key:
  ```bash
  $env:ANTHROPIC_API_KEY = "sk-..."   # PowerShell
  export ANTHROPIC_API_KEY="sk-..."   # bash
  ```
- **`05_benchmark.py`** — times every *installed* engine and overlays each
  profiler's recommendation on the chart.
  - `duckdb`/`polars`/`dask` come from the `examples` group already; any
    engine whose package isn't installed is skipped automatically, and the
    benchmark always runs with at least pandas.
  - Spark is optional and detected the same way — install it separately if
    you want it included:
    ```bash
    uv sync --no-default-groups --group examples --extra spark   # needs Java 8+
    ```
  - Unlike the engine detection, the LLM profiler section is **not** a
    graceful skip: if `anthropic` is importable (it is, via the `examples`
    group) but `ANTHROPIC_API_KEY` isn't set, the script raises instead of
    skipping. Set the key first, or comment out that section.
- **`06_format_profiler.py`** — `RuleBasedFormatProfiler` recommending an
  output *file format* (not engine) from size, operations, schema, and
  write-mode heuristics; no extra setup.
- **`07_resource_profiler.py`** — `RuleBasedResourceProfiler` sizing a
  cores/memory budget against a fixed 16-core/64 GB environment; no extra
  setup.
- **`08_partition_profiler.py`** — `RuleBasedPartitionProfiler` picking a
  partition column from schema heuristics, then blended with a stub "LLM"
  client via `build_ensemble_partition_profiler` (no API key needed — the
  stub stands in for `AnthropicLLMClient`/`OpenAILLMClient`, same idea as
  `04_llm_prefetch.py` but without a live call).
- **`09_cost_based_engine_profiler.py`** — `CostBasedEngineProfiler` picking
  the cheapest engine at illustrative $/hr rates; no extra setup.

## Everything at once

```bash
uv sync --no-default-groups --group examples --extra spark
```
