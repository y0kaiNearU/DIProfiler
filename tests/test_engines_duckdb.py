import narwhals as nw
import pytest

from engines.duckdb.loader import DuckDBLoader
from engines.duckdb.writer import DuckDBWriter
from models.models import DatasetInfo, FileFormat, FileSource, PipelineRequest, WriteMode


def _req(src_path, src_fmt, dst_path=None, dst_fmt=None, write_mode=WriteMode.OVERWRITE):
    destination = None
    if dst_path is not None:
        destination = DatasetInfo(source=FileSource(path=dst_path, format=dst_fmt, write_mode=write_mode))
    return PipelineRequest(
        source=DatasetInfo(source=FileSource(path=src_path, format=src_fmt)),
        destination=destination,
    )


def _write_file(path, fmt, rows):
    if fmt == FileFormat.CSV:
        path.write_text("a,b\n" + "\n".join(f"{a},{b}" for a, b in rows))
    elif fmt == FileFormat.JSON:
        path.write_text("\n".join(f'{{"a": {a}, "b": "{b}"}}' for a, b in rows))


class TestDuckDBLoader:
    def test_can_load_true_for_supported_format(self):
        req = _req("x.csv", FileFormat.CSV)
        assert DuckDBLoader().can_load(req)

    def test_can_load_false_for_unsupported_format(self):
        req = _req("x.orc", FileFormat.ORC)
        assert not DuckDBLoader().can_load(req)

    def test_load_csv(self, tmp_path):
        src = tmp_path / "in.csv"
        _write_file(src, FileFormat.CSV, [(1, "x"), (2, "y")])
        req = _req(str(src), FileFormat.CSV)

        frame = DuckDBLoader().load(req)
        rows = frame.collect().rows(named=True)
        assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_load_json(self, tmp_path):
        src = tmp_path / "in.json"
        _write_file(src, FileFormat.JSON, [(1, "x"), (2, "y")])
        req = _req(str(src), FileFormat.JSON)

        frame = DuckDBLoader().load(req)
        rows = frame.collect().rows(named=True)
        assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


class TestDuckDBWriter:
    def test_can_write_false_without_destination(self):
        req = _req("x.csv", FileFormat.CSV)
        assert not DuckDBWriter().can_write(req)

    def test_can_write_false_for_unsupported_format(self):
        req = _req("x.csv", FileFormat.CSV, "y.orc", FileFormat.ORC)
        assert not DuckDBWriter().can_write(req)

    @pytest.mark.parametrize("fmt", [FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON])
    def test_round_trip(self, tmp_path, fmt):
        src = tmp_path / "in.csv"
        _write_file(src, FileFormat.CSV, [(1, "x"), (2, "y")])
        dst = tmp_path / f"out.{fmt.value}"
        req = _req(str(src), FileFormat.CSV, str(dst), fmt)

        frame = DuckDBLoader().load(req)
        DuckDBWriter().write(frame, req)

        assert dst.exists()
        result = DuckDBLoader().load(_req(str(dst), fmt))
        rows = sorted(result.collect().rows(named=True), key=lambda r: r["a"])
        assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_append_adds_rows(self, tmp_path):
        src = tmp_path / "in.csv"
        _write_file(src, FileFormat.CSV, [(1, "x"), (2, "y")])
        dst = tmp_path / "out.csv"

        first_req = _req(str(src), FileFormat.CSV, str(dst), FileFormat.CSV)
        DuckDBWriter().write(DuckDBLoader().load(first_req), first_req)

        second_src = tmp_path / "more.csv"
        _write_file(second_src, FileFormat.CSV, [(3, "z")])
        append_req = _req(str(second_src), FileFormat.CSV, str(dst), FileFormat.CSV, write_mode=WriteMode.APPEND)
        DuckDBWriter().write(DuckDBLoader().load(append_req), append_req)

        result = DuckDBLoader().load(_req(str(dst), FileFormat.CSV))
        rows = sorted(result.collect().rows(named=True), key=lambda r: r["a"])
        assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
