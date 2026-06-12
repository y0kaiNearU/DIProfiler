from __future__ import annotations

import narwhals as nw

from core.loader import Loader
from models.models import DatabaseSource, EngineType, FileFormat, FileSource, PipelineRequest


class DuckDBLoader(Loader):

    @property
    def engine(self) -> EngineType:
        return EngineType.DUCKDB

    def can_load(self, request: PipelineRequest) -> bool:
        src = request.source.source
        if isinstance(src, FileSource):
            return src.format in (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)
        elif isinstance(src, DatabaseSource):
            return src.database_type == "postgresql"
        return False

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        try:
            import duckdb
        except ImportError as e:
            raise ImportError("DuckDB is required: uv add duckdb") from e

        src = request.source.source
        conn = duckdb.connect()

        if isinstance(src, FileSource):
            match src.format:
                case FileFormat.CSV:
                    native = conn.read_csv(src.path)
                case FileFormat.PARQUET:
                    native = conn.read_parquet(src.path)
                case FileFormat.JSON:
                    native = conn.read_json(src.path)
                case _:
                    raise NotImplementedError(f"DuckDB loader does not support {src.format}")
        elif isinstance(src, DatabaseSource):
            if src.database_type == "postgresql":
                conn.install_extension("postgres_scanner")
                conn.load_extension("postgres_scanner")
                native = conn.execute(
                    f"SELECT * FROM postgres_scan('{src.connection_string}', '{src.table_name}')"
                ).fetch_arrow_table()
            else:
                raise NotImplementedError(f"DuckDB loader does not support {src.database_type}")
        else:
            raise ValueError(f"Unknown source type: {type(src)}")

        return nw.from_native(native)

