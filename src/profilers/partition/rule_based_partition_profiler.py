from __future__ import annotations

from core.profiler import Profiler
from models.models import PartitionRecommendation, PipelineRequest, ProfilingResult
from profilers.common.voting import Rule as _Rule
from profilers.common.voting import Vote as _Vote
from profilers.common.voting import aggregate_votes

Vote = _Vote[str]
Rule = _Rule[str]

_TEMPORAL_NAME_HINTS = ("date", "time", "day", "month", "year", "_dt")
_TEMPORAL_TYPE_HINTS = ("date", "timestamp", "datetime")
_CATEGORICAL_NAME_HINTS = ("status", "type", "category", "region", "country", "state")
_LOW_CARDINALITY_TYPES = ("bool", "boolean", "category", "enum")


def _is_identifier_like(column: str) -> bool:
    lowered = column.lower()
    return lowered == "id" or lowered.endswith("_id") or "uuid" in lowered or "guid" in lowered


def _temporal_name_rule(column: str) -> Rule:
    def rule(req: PipelineRequest) -> Vote | None:
        lowered = column.lower()
        if any(hint in lowered for hint in _TEMPORAL_NAME_HINTS):
            return column, 0.7, f"column name '{column}' suggests a temporal field, a common partition key"
        return None
    return rule


def _temporal_type_rule(column: str, col_type: str) -> Rule:
    def rule(req: PipelineRequest) -> Vote | None:
        lowered = col_type.lower()
        if any(hint in lowered for hint in _TEMPORAL_TYPE_HINTS):
            return column, 0.6, f"column '{column}' has type '{col_type}', typically used for time-based partitioning"
        return None
    return rule


def _categorical_name_rule(column: str) -> Rule:
    def rule(req: PipelineRequest) -> Vote | None:
        lowered = column.lower()
        if any(hint in lowered for hint in _CATEGORICAL_NAME_HINTS):
            return column, 0.5, f"column name '{column}' suggests a low-cardinality categorical field"
        return None
    return rule


def _low_cardinality_type_rule(column: str, col_type: str) -> Rule:
    def rule(req: PipelineRequest) -> Vote | None:
        lowered = col_type.lower()
        if lowered in _LOW_CARDINALITY_TYPES:
            return column, 0.4, f"column '{column}' has type '{col_type}', a low-cardinality partitioning key"
        return None
    return rule


def _build_rules(schema: dict[str, str]) -> list[Rule]:
    """One closure per (column, heuristic) so each column can be scored independently."""
    rules: list[Rule] = []
    for column, col_type in schema.items():
        if _is_identifier_like(column):
            continue
        rules.append(_temporal_name_rule(column))
        rules.append(_temporal_type_rule(column, col_type))
        rules.append(_categorical_name_rule(column))
        rules.append(_low_cardinality_type_rule(column, col_type))
    return rules


class RuleBasedPartitionProfiler(Profiler[PartitionRecommendation]):
    """
    Recommends a partition column from PipelineRequest.source.schema using
    name/type heuristics (temporal and low-cardinality categorical columns
    are favoured; identifier-like columns are never recommended).
    """

    @property
    def name(self) -> str:
        return "rule_based_partition_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return bool(request.source.schema)

    def profile(self, request: PipelineRequest) -> ProfilingResult[PartitionRecommendation]:
        schema = request.source.schema
        scored = aggregate_votes(request, _build_rules(schema), set(schema))

        recommendations = [
            PartitionRecommendation(column=column, confidence=confidence, reasoning=reasoning)
            for column, confidence, reasoning in scored
        ]

        return ProfilingResult(request=request, recommendations=recommendations)
