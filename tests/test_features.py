import math

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
from profilers.features import FEATURE_NAMES, extract


def _req(
    fmt=FileFormat.CSV,
    size_bytes=None,
    row_count=None,
    num_columns=None,
    ops=None,
    dst_fmt=None,
    db_type=None,
):
    if db_type:
        src = DatabaseSource(
            connection_string="postgresql://localhost/db",
            table_name="t",
            database_type=db_type,
        )
    else:
        src = FileSource(path="x", format=fmt)

    dst_info = None
    if dst_fmt:
        dst_info = DatasetInfo(source=FileSource(path="out", format=dst_fmt))

    return PipelineRequest(
        source=DatasetInfo(source=src, size_bytes=size_bytes, row_count=row_count, num_columns=num_columns),
        operations=ops or [],
        destination=dst_info,
        available_engines=list(EngineType),
    )


def test_feature_length_matches_feature_names():
    feats = extract(_req())
    assert len(feats) == len(FEATURE_NAMES)


def test_log_size_bytes_correct():
    feats = extract(_req(size_bytes=1000))
    idx = FEATURE_NAMES.index("log_size_bytes")
    assert feats[idx] == pytest.approx(math.log1p(1000))


def test_log_size_bytes_zero_when_none():
    feats = extract(_req(size_bytes=None))
    idx = FEATURE_NAMES.index("log_size_bytes")
    assert feats[idx] == 0.0


def test_log_row_count_correct():
    feats = extract(_req(row_count=5000))
    idx = FEATURE_NAMES.index("log_row_count")
    assert feats[idx] == pytest.approx(math.log1p(5000))


def test_num_columns():
    feats = extract(_req(num_columns=42))
    idx = FEATURE_NAMES.index("num_columns")
    assert feats[idx] == 42.0


def test_num_columns_zero_when_none():
    feats = extract(_req(num_columns=None))
    idx = FEATURE_NAMES.index("num_columns")
    assert feats[idx] == 0.0


@pytest.mark.parametrize("fmt", list(FileFormat))
def test_source_format_one_hot(fmt):
    feats = extract(_req(fmt=fmt))
    for f in FileFormat:
        idx = FEATURE_NAMES.index(f"src_fmt_{f.value}")
        expected = 1.0 if f == fmt else 0.0
        assert feats[idx] == expected, f"src_fmt_{f.value} mismatch for format {fmt}"


def test_src_is_file_for_file_source():
    feats = extract(_req(fmt=FileFormat.CSV))
    assert feats[FEATURE_NAMES.index("src_is_file")] == 1.0
    assert feats[FEATURE_NAMES.index("src_is_database")] == 0.0


def test_src_is_database_for_database_source():
    feats = extract(_req(db_type="postgresql"))
    assert feats[FEATURE_NAMES.index("src_is_file")] == 0.0
    assert feats[FEATURE_NAMES.index("src_is_database")] == 1.0


def test_src_db_type_one_hot():
    feats = extract(_req(db_type="mysql"))
    assert feats[FEATURE_NAMES.index("src_db_mysql")] == 1.0
    assert feats[FEATURE_NAMES.index("src_db_postgresql")] == 0.0


def test_has_destination_zero_when_none():
    feats = extract(_req())
    assert feats[FEATURE_NAMES.index("has_destination")] == 0.0


def test_has_destination_one_when_set():
    feats = extract(_req(dst_fmt=FileFormat.PARQUET))
    assert feats[FEATURE_NAMES.index("has_destination")] == 1.0


def test_destination_format_one_hot():
    feats = extract(_req(dst_fmt=FileFormat.PARQUET))
    assert feats[FEATURE_NAMES.index("dst_fmt_parquet")] == 1.0
    assert feats[FEATURE_NAMES.index("dst_fmt_csv")] == 0.0


def test_operation_flags():
    feats = extract(_req(ops=[OperationType.JOIN, OperationType.WINDOW]))
    assert feats[FEATURE_NAMES.index("op_join")] == 1.0
    assert feats[FEATURE_NAMES.index("op_window")] == 1.0
    assert feats[FEATURE_NAMES.index("op_filter")] == 0.0


def test_no_operations_all_zero():
    feats = extract(_req(ops=[]))
    for op in OperationType:
        assert feats[FEATURE_NAMES.index(f"op_{op.value}")] == 0.0
