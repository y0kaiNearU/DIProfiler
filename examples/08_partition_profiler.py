import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt

from models.models import DatasetInfo, FileFormat, FileSource, PipelineRequest
from profilers.partition.ensemble_partition_profiler import build_ensemble_partition_profiler
from profilers.partition.rule_based_partition_profiler import RuleBasedPartitionProfiler

_GB = 1024 ** 3

_SCHEMA = {
    "transaction_id": "string",
    "event_date": "date",
    "region": "string",
    "status": "category",
    "amount": "float",
}

request = PipelineRequest(
    source=DatasetInfo(
        source=FileSource(path="data/transactions.parquet", format=FileFormat.PARQUET),
        size_bytes=40 * _GB,
        row_count=500_000_000,
        schema=_SCHEMA,
    ),
)


# ---------------------------------------------------------------------------
# Step 1 - Rule-based partition profiler alone (name/type heuristics only)
# ---------------------------------------------------------------------------

rule_profiler = RuleBasedPartitionProfiler()
rule_result = rule_profiler.profile(request)

print("Rule-based partition profiler")
print("-" * 40)
for rec in rule_result.recommendations:
    marker = " <-- best" if rec == rule_result.best else ""
    print(f"  {rec.column:<15} confidence={rec.confidence:.3f}  {rec.reasoning}{marker}")
print("  (transaction_id excluded: identifier-like columns are never recommended)")
print("  (amount excluded: no name/type heuristic matched -> zero votes)")


# ---------------------------------------------------------------------------
# Step 2 - A real LLM client if ANTHROPIC_API_KEY is set, else a stub
#
# LLMPartitionClient is a Protocol (see llm_partition_profiler.py): anything
# with a .recommend(prompt, column_options) -> list[dict] method works. When
# a key is available we use the real Claude-backed client (same one
# examples/04_llm_prefetch.py calls); otherwise we fall back to a stub that
# returns canned scores, so this example always runs without requiring an
# API key or network access.
# ---------------------------------------------------------------------------

class StubLLMClient:
    """Pretends to be an LLM with a semantic read on the schema: it favours
    'region' as a low-cardinality analytical dimension more strongly than the
    rule-based profiler's name-matching does, and is skeptical of 'status'."""

    def recommend(self, prompt: str, column_options: list[str]) -> list[dict[str, Any]]:
        scores = {
            "event_date": (0.55, "time-based partitioning is standard for append-heavy transaction tables"),
            "region": (0.8, "low-cardinality geographic dimension, ideal for partition pruning in analytical queries"),
            "status": (0.25, "low cardinality but rarely a useful partition key for time-series analytics"),
        }
        return [
            {"column": col, "confidence": conf, "reasoning": reason}
            for col, (conf, reason) in scores.items()
            if col in column_options
        ]


if os.environ.get("ANTHROPIC_API_KEY"):
    from profilers.partition.llm_clients.anthropic_client import AnthropicLLMClient

    print("\nANTHROPIC_API_KEY found -> using a real Claude-backed LLM client")
    llm_client: Any = AnthropicLLMClient()
else:
    print("\nANTHROPIC_API_KEY not set -> using a stub LLM client (no network call)")
    llm_client = StubLLMClient()

ensemble_profiler = build_ensemble_partition_profiler(llm_client=llm_client)
ensemble_result = ensemble_profiler.profile(request)

print("\nEnsemble (rule-based + LLM, equal weights)")
print("-" * 40)
for rec in ensemble_result.recommendations:
    marker = " <-- best" if rec == ensemble_result.best else ""
    print(f"  {rec.column:<15} confidence={rec.confidence:.3f}{marker}")
    print(f"                  {rec.reasoning}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

fig, (ax_rule, ax_ensemble) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Partition profiler — rule-based alone vs. ensembled with an LLM", fontsize=12, fontweight="bold")

for ax, result, title in (
    (ax_rule, rule_result, "Rule-based only"),
    (ax_ensemble, ensemble_result, "Rule-based + LLM ensemble"),
):
    columns = [rec.column for rec in result.recommendations]
    confs = [rec.confidence for rec in result.recommendations]
    bars = ax.barh(columns, confs, color="#7b4fd6", edgecolor="white", height=0.5)
    ax.set_xlim(0, max(confs, default=1.0) * 1.3)
    ax.set_xlabel("Confidence", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    for bar, conf in zip(bars, confs):
        ax.text(conf + 0.01, bar.get_y() + bar.get_height() / 2, f"{conf:.2f}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
out = Path(__file__).parent / "08_partition_profiler_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved -> {out}")
plt.show()
