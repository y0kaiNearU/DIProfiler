import pytest

from models.models import (
    DatasetInfo,
    DatabaseSource,
    EngineRecommendation,
    EngineType,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
    ProfilingResult,
    WriteMode,
)


def test_file_source_defaults():
    src = FileSource(path="data.csv", format=FileFormat.CSV)
    assert src.write_mode == WriteMode.OVERWRITE


def test_database_source_defaults():
    src = DatabaseSource(
        connection_string="postgresql://localhost/db",
        table_name="events",
        database_type="postgresql",
    )
    assert src.schema == "public"
    assert src.queries == {}
    assert src.write_mode == WriteMode.OVERWRITE


def test_dataset_info_optional_fields():
    src = FileSource(path="x.parquet", format=FileFormat.PARQUET)
    info = DatasetInfo(source=src)
    assert info.size_bytes is None
    assert info.row_count is None
    assert info.num_columns is None
    assert info.schema == {}


def test_dataset_info_with_all_fields():
    src = FileSource(path="x.csv", format=FileFormat.CSV)
    info = DatasetInfo(
        source=src,
        size_bytes=1024,
        row_count=100,
        num_columns=5,
        schema={"id": "int64", "name": "str"},
    )
    assert info.size_bytes == 1024
    assert info.row_count == 100
    assert info.num_columns == 5
    assert info.schema["id"] == "int64"


def test_pipeline_request_defaults():
    src = DatasetInfo(source=FileSource(path="x.csv", format=FileFormat.CSV))
    req = PipelineRequest(source=src)
    assert req.available_engines == list(EngineType)
    assert req.operations == []
    assert req.destination is None
    assert req.required_engine is None


def test_pipeline_request_custom_engines():
    src = DatasetInfo(source=FileSource(path="x.csv", format=FileFormat.CSV))
    req = PipelineRequest(
        source=src,
        available_engines=[EngineType.DUCKDB],
        operations=[OperationType.FILTER],
    )
    assert req.available_engines == [EngineType.DUCKDB]
    assert OperationType.FILTER in req.operations


def _make_request():
    return PipelineRequest(
        source=DatasetInfo(source=FileSource(path="x.csv", format=FileFormat.CSV))
    )


def test_profiling_result_best_returns_highest_confidence():
    req = _make_request()
    recs = [
        EngineRecommendation(engine=EngineType.DUCKDB, confidence=0.6, reasoning="a"),
        EngineRecommendation(engine=EngineType.SPARK, confidence=0.3, reasoning="b"),
        EngineRecommendation(engine=EngineType.DATAFUSION, confidence=0.1, reasoning="c"),
    ]
    result = ProfilingResult(request=req, recommendations=recs)
    assert result.best.engine == EngineType.DUCKDB
    assert result.best.confidence == 0.6


def test_profiling_result_best_returns_none_when_empty():
    result = ProfilingResult(request=_make_request(), recommendations=[])
    assert result.best is None


def test_profiling_result_best_single_recommendation():
    req = _make_request()
    rec = EngineRecommendation(engine=EngineType.SPARK, confidence=1.0, reasoning="only one")
    result = ProfilingResult(request=req, recommendations=[rec])
    assert result.best == rec
