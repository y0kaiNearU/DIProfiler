import pytest

from models.models import DatabaseSource, DatasetInfo, FileFormat, FileSource, OperationType, PipelineRequest
from profilers.partition.llm_partition_profiler import LLMPartitionProfiler, _format_request


def _req(
    schema=None,
    size_bytes=None,
    row_count=None,
    ops=None,
    db_type=None,
):
    if db_type:
        src = DatabaseSource(
            connection_string="conn",
            table_name="tbl",
            database_type=db_type,
        )
    else:
        src = FileSource(path="data/x.parquet", format=FileFormat.PARQUET)

    return PipelineRequest(
        source=DatasetInfo(
            source=src,
            size_bytes=size_bytes,
            row_count=row_count,
            schema=schema or {},
        ),
        operations=ops or [],
    )


class _StubLLMClient:
    """A minimal LLMPartitionClient for testing LLMPartitionProfiler without any real provider."""

    def __init__(self, recommendations):
        self._recommendations = recommendations
        self.last_prompt = None
        self.last_column_options = None

    def recommend(self, prompt, column_options):
        self.last_prompt = prompt
        self.last_column_options = column_options
        return self._recommendations


def test_format_request_includes_file_path_and_format():
    text = _format_request(_req(schema={"created_date": "date"}))
    assert "parquet" in text
    assert "data/x.parquet" in text


def test_format_request_includes_size_when_set():
    text = _format_request(_req(size_bytes=2 * 1024 ** 3))
    assert "2.000 GB" in text


def test_format_request_omits_size_when_none():
    text = _format_request(_req(size_bytes=None))
    assert "Size" not in text


def test_format_request_includes_operations():
    text = _format_request(_req(ops=[OperationType.JOIN, OperationType.WINDOW]))
    assert "join" in text
    assert "window" in text


def test_format_request_includes_schema_and_candidate_columns():
    text = _format_request(_req(schema={"created_date": "date", "user_id": "int"}))
    assert "created_date" in text
    assert "user_id" in text
    assert "Candidate partition columns" in text


def test_format_request_database_source():
    text = _format_request(_req(db_type="postgresql"))
    assert "postgresql" in text
    assert "tbl" in text


def test_name():
    assert LLMPartitionProfiler(_StubLLMClient([])).name == "llm_partition_profiler"


def test_can_handle_true_when_schema_present():
    profiler = LLMPartitionProfiler(_StubLLMClient([]))
    assert profiler.can_handle(_req(schema={"created_date": "date"}))


def test_can_handle_false_when_schema_empty():
    profiler = LLMPartitionProfiler(_StubLLMClient([]))
    assert not profiler.can_handle(_req(schema={}))


def test_default_client_is_anthropic():
    from profilers.partition.llm_clients.anthropic_client import AnthropicLLMClient

    profiler = LLMPartitionProfiler()
    assert isinstance(profiler._get_client(), AnthropicLLMClient)


def test_profile_returns_correct_recommendations():
    client = _StubLLMClient([
        {"column": "created_date", "confidence": 0.7, "reasoning": "temporal, good for pruning"},
        {"column": "status", "confidence": 0.3, "reasoning": "low cardinality"},
    ])
    profiler = LLMPartitionProfiler(client)

    req = _req(schema={"created_date": "date", "status": "string", "user_id": "int"})
    result = profiler.profile(req)
    assert result.best.column == "created_date"
    assert result.best.confidence == pytest.approx(0.7)


def test_profile_recommendations_sorted_descending():
    client = _StubLLMClient([
        {"column": "status", "confidence": 0.3, "reasoning": "r"},
        {"column": "created_date", "confidence": 0.7, "reasoning": "r"},
    ])
    profiler = LLMPartitionProfiler(client)

    req = _req(schema={"created_date": "date", "status": "string"})
    result = profiler.profile(req)
    confidences = [r.confidence for r in result.recommendations]
    assert confidences == sorted(confidences, reverse=True)


def test_profile_filters_columns_not_in_schema():
    client = _StubLLMClient([
        {"column": "created_date", "confidence": 0.8, "reasoning": "r"},
        {"column": "not_a_real_column", "confidence": 0.2, "reasoning": "r"},
    ])
    profiler = LLMPartitionProfiler(client)

    req = _req(schema={"created_date": "date"})
    result = profiler.profile(req)

    columns = {r.column for r in result.recommendations}
    assert columns == {"created_date"}


def test_profile_passes_schema_columns_to_client():
    client = _StubLLMClient([{"column": "created_date", "confidence": 1.0, "reasoning": "r"}])
    profiler = LLMPartitionProfiler(client)

    req = _req(schema={"created_date": "date", "user_id": "int"})
    profiler.profile(req)

    assert set(client.last_column_options) == {"created_date", "user_id"}


def test_profile_passes_formatted_prompt_to_client():
    client = _StubLLMClient([{"column": "created_date", "confidence": 1.0, "reasoning": "r"}])
    profiler = LLMPartitionProfiler(client)

    req = _req(schema={"created_date": "date"})
    profiler.profile(req)

    assert client.last_prompt == _format_request(req)
