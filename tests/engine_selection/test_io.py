import pytest

from engine_selection.io import FrameLoader, FrameWriter
from engine_selection.loader import Loader
from engine_selection.writer import Writer
from models.models import DatabaseSource, DatasetInfo, EngineType, FileFormat, FileSource, PipelineRequest


class _StubLoader(Loader):
    def __init__(self, engine, can_load=True, result="loaded"):
        self._engine = engine
        self._can_load = can_load
        self._result = result

    @property
    def engine(self):
        return self._engine

    def can_load(self, request):
        return self._can_load

    def load(self, request):
        return self._result


class _StubWriter(Writer):
    def __init__(self, engine, can_write=True):
        self._engine = engine
        self._can_write = can_write
        self.written = None

    @property
    def engine(self):
        return self._engine

    def can_write(self, request):
        return self._can_write

    def write(self, frame, request):
        self.written = (frame, request)


def _req(destination=None):
    return PipelineRequest(
        source=DatasetInfo(source=FileSource(path="x.csv", format=FileFormat.CSV)),
        destination=destination,
    )


class TestFrameLoader:
    def test_load_resolves_correct_engine_loader(self):
        loader = _StubLoader(EngineType.DUCKDB)
        frame_loader = FrameLoader(EngineType.DUCKDB, loaders=[loader])

        assert frame_loader.load(_req()) == "loaded"

    def test_load_raises_not_implemented_when_no_loader_can_handle(self):
        loader = _StubLoader(EngineType.DUCKDB, can_load=False)
        frame_loader = FrameLoader(EngineType.DUCKDB, loaders=[loader])

        with pytest.raises(NotImplementedError):
            frame_loader.load(_req())

    def test_load_works_for_database_source(self):
        # Regression: DatabaseSource has no `.format` attribute; the error path
        # must not assume the source is a FileSource.
        loader = _StubLoader(EngineType.DUCKDB, can_load=False)
        frame_loader = FrameLoader(EngineType.DUCKDB, loaders=[loader])
        req = PipelineRequest(
            source=DatasetInfo(
                source=DatabaseSource(connection_string="c", table_name="t", database_type="postgresql")
            ),
        )

        with pytest.raises(NotImplementedError):
            frame_loader.load(req)


class TestFrameWriter:
    def test_write_raises_when_no_destination(self):
        writer = _StubWriter(EngineType.DUCKDB)
        frame_writer = FrameWriter(EngineType.DUCKDB, writers=[writer])

        with pytest.raises(ValueError):
            frame_writer.write("frame", _req())

    def test_write_resolves_correct_engine_writer(self):
        writer = _StubWriter(EngineType.DUCKDB)
        frame_writer = FrameWriter(EngineType.DUCKDB, writers=[writer])
        dest = DatasetInfo(source=FileSource(path="out.csv", format=FileFormat.CSV))
        req = _req(destination=dest)

        frame_writer.write("frame", req)

        assert writer.written == ("frame", req)

    def test_write_raises_not_implemented_when_no_writer_can_handle(self):
        writer = _StubWriter(EngineType.DUCKDB, can_write=False)
        frame_writer = FrameWriter(EngineType.DUCKDB, writers=[writer])
        dest = DatasetInfo(source=FileSource(path="out.csv", format=FileFormat.CSV))

        with pytest.raises(NotImplementedError):
            frame_writer.write("frame", _req(destination=dest))
