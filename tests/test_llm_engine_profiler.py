import pytest

from models.models import (
    DatasetInfo,
    DatabaseSource,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
)
from profilers.engine.llm_engine_profiler import LLMEngineProfiler, _format_request


def _req(
    fmt=FileFormat.CSV,
    size_bytes=None,
    row_count=None,
    num_columns=None,
    ops=None,
    db_type=None,
    available_engines=None,
):
    if db_type:
        src = DatabaseSource(
            connection_string="conn",
            table_name="tbl",
            database_type=db_type,
        )
    else:
        src = FileSource(path="data/x.parquet", format=fmt)

    return PipelineRequest(
        source=DatasetInfo(
            source=src,
            size_bytes=size_bytes,
            row_count=row_count,
            num_columns=num_columns,
        ),
        operations=ops or [],
        available_engines=available_engines if available_engines is not None else list(EngineType),
    )


class _StubLLMClient:
    """A minimal LLMEngineClient for testing LLMEngineProfiler without any real provider."""

    def __init__(self, recommendations):
        self._recommendations = recommendations
        self.last_prompt = None
        self.last_engine_options = None

    def recommend(self, prompt, engine_options):
        self.last_prompt = prompt
        self.last_engine_options = engine_options
        return self._recommendations


def test_format_request_includes_file_path_and_format():
    text = _format_request(_req(fmt=FileFormat.PARQUET))
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


def test_format_request_includes_available_engines():
    text = _format_request(_req(available_engines=[EngineType.DUCKDB]))
    assert "duckdb" in text


def test_format_request_database_source():
    text = _format_request(_req(db_type="postgresql"))
    assert "postgresql" in text
    assert "tbl" in text


def test_name():
    assert LLMEngineProfiler(_StubLLMClient([])).name == "llm_engine_profiler"


def test_can_handle_always_true():
    profiler = LLMEngineProfiler(_StubLLMClient([]))
    assert profiler.can_handle(_req())


def test_default_client_is_anthropic():
    from profilers.engine.anthropic_llm_client import AnthropicLLMClient

    profiler = LLMEngineProfiler()
    assert isinstance(profiler._get_client(), AnthropicLLMClient)


def test_profile_returns_correct_recommendations():
    client = _StubLLMClient([
        {"engine": "duckdb", "confidence": 0.7, "reasoning": "small file"},
        {"engine": "spark", "confidence": 0.2, "reasoning": "overkill but available"},
        {"engine": "datafusion", "confidence": 0.1, "reasoning": "viable"},
    ])
    profiler = LLMEngineProfiler(client)

    result = profiler.profile(_req())
    assert result.best.engine == EngineType.DUCKDB
    assert result.best.confidence == pytest.approx(0.7)


def test_profile_recommendations_sorted_descending():
    client = _StubLLMClient([
        {"engine": "spark", "confidence": 0.5, "reasoning": "r"},
        {"engine": "duckdb", "confidence": 0.3, "reasoning": "r"},
        {"engine": "datafusion", "confidence": 0.2, "reasoning": "r"},
    ])
    profiler = LLMEngineProfiler(client)

    result = profiler.profile(_req())
    confidences = [r.confidence for r in result.recommendations]
    assert confidences == sorted(confidences, reverse=True)


def test_profile_filters_unavailable_engines():
    client = _StubLLMClient([
        {"engine": "duckdb", "confidence": 0.8, "reasoning": "r"},
        {"engine": "spark", "confidence": 0.2, "reasoning": "r"},
    ])
    profiler = LLMEngineProfiler(client)

    req = _req(available_engines=[EngineType.DUCKDB])
    result = profiler.profile(req)

    engines = {r.engine for r in result.recommendations}
    assert EngineType.SPARK not in engines
    assert EngineType.DUCKDB in engines


def test_profile_passes_available_engines_to_client():
    client = _StubLLMClient([{"engine": "duckdb", "confidence": 1.0, "reasoning": "r"}])
    profiler = LLMEngineProfiler(client)

    profiler.profile(_req(available_engines=[EngineType.DUCKDB, EngineType.SPARK]))

    assert set(client.last_engine_options) == {"duckdb", "spark"}


def test_profile_passes_formatted_prompt_to_client():
    client = _StubLLMClient([{"engine": "duckdb", "confidence": 1.0, "reasoning": "r"}])
    profiler = LLMEngineProfiler(client)

    req = _req(fmt=FileFormat.PARQUET)
    profiler.profile(req)

    assert client.last_prompt == _format_request(req)
