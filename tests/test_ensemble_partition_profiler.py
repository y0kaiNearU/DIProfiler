import pytest

from models.models import DatasetInfo, FileFormat, FileSource, PipelineRequest
from profilers.partition.ensemble_partition_profiler import build_ensemble_partition_profiler
from profilers.partition.llm_partition_profiler import LLMPartitionProfiler
from profilers.partition.rule_based_partition_profiler import RuleBasedPartitionProfiler


def _req(schema=None):
    return PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="x", format=FileFormat.CSV),
            schema=schema or {},
        )
    )


class _StubLLMClient:
    def __init__(self, recommendations):
        self._recommendations = recommendations

    def recommend(self, prompt, column_options):
        return self._recommendations


def test_combines_rule_based_and_llm_profilers():
    schema = {"created_date": "date", "status": "string", "user_id": "int"}
    llm_client = _StubLLMClient([
        {"column": "created_date", "confidence": 0.9, "reasoning": "llm likes created_date"},
    ])
    profiler = build_ensemble_partition_profiler(llm_client=llm_client)

    result = profiler.profile(_req(schema=schema))

    assert result.best is not None
    assert result.best.column == "created_date"
    # both profilers voted for created_date (rule-based via temporal heuristics, llm directly)
    assert "llm likes created_date" in result.best.reasoning


def test_can_handle_true_when_schema_present():
    profiler = build_ensemble_partition_profiler(llm_client=_StubLLMClient([]))
    assert profiler.can_handle(_req(schema={"created_date": "date"}))


def test_can_handle_false_when_schema_empty():
    profiler = build_ensemble_partition_profiler(llm_client=_StubLLMClient([]))
    assert not profiler.can_handle(_req(schema={}))


def test_confidences_sum_to_one():
    schema = {"created_date": "date", "status": "string"}
    llm_client = _StubLLMClient([
        {"column": "created_date", "confidence": 0.6, "reasoning": "r"},
        {"column": "status", "confidence": 0.4, "reasoning": "r"},
    ])
    profiler = build_ensemble_partition_profiler(llm_client=llm_client)

    result = profiler.profile(_req(schema=schema))
    total = sum(r.confidence for r in result.recommendations)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_weights_shift_balance_toward_llm():
    schema = {"created_date": "date", "region": "string"}
    llm_client = _StubLLMClient([
        {"column": "region", "confidence": 1.0, "reasoning": "llm strongly prefers region"},
    ])

    equal = build_ensemble_partition_profiler(llm_client=llm_client)
    llm_heavy = build_ensemble_partition_profiler(llm_client=llm_client, weights=[0.1, 0.9])

    equal_region = next(r.confidence for r in equal.profile(_req(schema=schema)).recommendations if r.column == "region")
    heavy_region = next(r.confidence for r in llm_heavy.profile(_req(schema=schema)).recommendations if r.column == "region")

    assert heavy_region > equal_region


def test_uses_rule_based_and_llm_profiler_instances():
    profiler = build_ensemble_partition_profiler(llm_client=_StubLLMClient([]))
    names = {p.name for p in profiler._profilers}
    assert RuleBasedPartitionProfiler().name in names
    assert LLMPartitionProfiler().name in names
