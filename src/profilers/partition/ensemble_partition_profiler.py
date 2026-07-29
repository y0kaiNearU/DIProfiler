from __future__ import annotations

from models.models import PartitionRecommendation
from profilers.common.composite import EnsembleProfiler
from profilers.partition.llm_partition_profiler import LLMPartitionClient, LLMPartitionProfiler
from profilers.partition.rule_based_partition_profiler import RuleBasedPartitionProfiler


def build_ensemble_partition_profiler(
    llm_client: LLMPartitionClient | None = None,
    weights: list[float] | None = None,
) -> EnsembleProfiler[PartitionRecommendation, str]:
    """
    Combines RuleBasedPartitionProfiler (name/type heuristics) and LLMPartitionProfiler
    (LLM judgment over the full schema + workload) into one ensemble, blending their
    confidences per candidate column.

    Args:
        llm_client: Passed through to LLMPartitionProfiler; defaults to AnthropicLLMClient.
        weights:    [rule_based_weight, llm_weight]. Defaults to equal weighting.
    """
    return EnsembleProfiler(
        [RuleBasedPartitionProfiler(), LLMPartitionProfiler(llm_client)],
        key_fn=lambda rec: rec.column,
        weights=weights,
        build_fn=lambda column, confidence, reasoning: PartitionRecommendation(
            column=column, confidence=confidence, reasoning=reasoning,
        ),
    )
