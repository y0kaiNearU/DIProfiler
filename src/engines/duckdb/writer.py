from __future__ import annotations

import narwhals as nw

from core.writer import Writer
from engines.duckdb.base import DuckDBBase
from engines.duckdb.database import writer as db_writer
from engines.duckdb.file import writer as file_writer
from models.models import DatabaseSource, EngineType, FileSource, PipelineRequest


class DuckDBWriter(DuckDBBase, Writer):

    @property
    def engine(self) -> EngineType:
        return EngineType.DUCKDB

    def can_write(self, request: PipelineRequest) -> bool:
        if request.destination is None:
            return False
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return dest.format in file_writer.SUPPORTED_FORMATS
        if isinstance(dest, DatabaseSource):
            return dest.database_type in db_writer.SUPPORTED_DATABASES
        return False

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return file_writer.write(self._get_connection(), frame, dest)
        if isinstance(dest, DatabaseSource):
            return db_writer.write(self._get_connection(), frame, dest)
        raise ValueError(f"Unknown destination type: {type(dest)}")
