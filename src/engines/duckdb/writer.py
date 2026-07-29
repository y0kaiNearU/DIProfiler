from __future__ import annotations

import narwhals as nw

from engine_selection.writer import Writer
from engines.duckdb import CAPABILITIES
from engines.duckdb.base import DuckDBBase
from engines.duckdb.database import writer as db_writer
from engines.duckdb.file import writer as file_writer
from models.models import DatabaseSource, EngineType, FileSource, PipelineRequest


class DuckDBWriter(DuckDBBase, Writer):
    capabilities = CAPABILITIES

    @property
    def engine(self) -> EngineType:
        return EngineType.DUCKDB

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        assert request.destination is not None
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return file_writer.write(self._get_connection(), frame, dest)
        if isinstance(dest, DatabaseSource):
            return db_writer.write(self._get_connection(), frame, dest)
        raise ValueError(f"Unknown destination type: {type(dest)}")
