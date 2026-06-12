from __future__ import annotations

import narwhals as nw

from core.loader import Loader
from engines.duckdb.base import DuckDBBase
from engines.duckdb.database import loader as db_loader
from engines.duckdb.file import loader as file_loader
from models.models import DatabaseSource, EngineType, FileSource, PipelineRequest


class DuckDBLoader(DuckDBBase, Loader):

    @property
    def engine(self) -> EngineType:
        return EngineType.DUCKDB

    def can_load(self, request: PipelineRequest) -> bool:
        src = request.source.source
        if isinstance(src, FileSource):
            return src.format in file_loader.SUPPORTED_FORMATS
        if isinstance(src, DatabaseSource):
            return src.database_type in db_loader.SUPPORTED_DATABASES
        return False

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(self._get_connection(), src)
        if isinstance(src, DatabaseSource):
            return db_loader.load(self._get_connection(), src)
        raise ValueError(f"Unknown source type: {type(src)}")
