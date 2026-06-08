from __future__ import annotations

import narwhals as nw

from core.loader import Loader
from models.models import EngineType, FileFormat, PipelineRequest


class DuckDBLoader(Loader):

    @property
    def engine(self) -> EngineType:
        return EngineType.DUCKDB

    def can_load(self, request: PipelineRequest) -> bool:
        return request.source.format in (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        try:
            import duckdb
        except ImportError as e:
            raise ImportError("DuckDB is required: uv add duckdb") from e

        conn = duckdb.connect()
        src = request.source
        match src.format:
            case FileFormat.CSV:
                native = conn.read_csv(src.path)
            case FileFormat.PARQUET:
                native = conn.read_parquet(src.path)
            case FileFormat.JSON:
                native = conn.read_json(src.path)
            case _:
                raise NotImplementedError(f"DuckDB loader does not support {src.format}")
        return nw.from_native(native)
