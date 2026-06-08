from __future__ import annotations

import narwhals as nw

from core.writer import Writer
from models.models import EngineType, FileFormat, PipelineRequest


class DuckDBWriter(Writer):

    @property
    def engine(self) -> EngineType:
        return EngineType.DUCKDB

    def can_write(self, request: PipelineRequest) -> bool:
        return (
            request.destination is not None
            and request.destination.format in (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)
        )

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        try:
            import duckdb
        except ImportError as e:
            raise ImportError("DuckDB is required: uv add duckdb") from e

        dest = request.destination
        rel = nw.to_native(frame)
        conn = duckdb.connect()
        conn.register("_frame", rel)
        match dest.format:
            case FileFormat.PARQUET:
                conn.execute(f"COPY _frame TO '{dest.path}' (FORMAT parquet)")
            case FileFormat.CSV:
                conn.execute(f"COPY _frame TO '{dest.path}' (FORMAT csv, HEADER TRUE)")
            case FileFormat.JSON:
                conn.execute(f"COPY _frame TO '{dest.path}' (FORMAT json)")
            case _:
                raise NotImplementedError(f"DuckDB writer does not support {dest.format}")
