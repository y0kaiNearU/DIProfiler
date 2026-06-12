from __future__ import annotations

import narwhals as nw

from core.writer import Writer
from models.models import DatabaseSource, EngineType, FileFormat, FileSource, PipelineRequest


class DuckDBWriter(Writer):

    @property
    def engine(self) -> EngineType:
        return EngineType.DUCKDB

    def can_write(self, request: PipelineRequest) -> bool:
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return dest.format in (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)
        elif isinstance(dest, DatabaseSource):
            return dest.database_type == "postgresql"
        return False

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        try:
            import duckdb
        except ImportError as e:
            raise ImportError("DuckDB is required: uv add duckdb") from e

        dest = request.destination.source
        rel = nw.to_native(frame)
        conn = duckdb.connect()
        conn.register("_frame", rel)

        if isinstance(dest, FileSource):
            match dest.format:
                case FileFormat.PARQUET:
                    conn.execute(f"COPY _frame TO '{dest.path}' (FORMAT parquet)")
                case FileFormat.CSV:
                    conn.execute(f"COPY _frame TO '{dest.path}' (FORMAT csv, HEADER TRUE)")
                case FileFormat.JSON:
                    conn.execute(f"COPY _frame TO '{dest.path}' (FORMAT json)")
                case _:
                    raise NotImplementedError(f"DuckDB writer does not support {dest.format}")
        elif isinstance(dest, DatabaseSource):
            match dest.database_type:
                case "postgresql":
                    conn.install_extension("postgres_scanner")
                    conn.load_extension("postgres_scanner")
                    conn.execute(
                        f"COPY _frame TO postgres('{dest.connection_string}', '{dest.table_name}')"
                    )
                case "mysql":
                    conn.install_extension("mysql")
                    conn.load_extension("mysql")
                    conn.execute(f"ATTACH '{dest.connection_string}' AS _mysql_db (TYPE mysql)")
                    conn.execute(f"INSERT INTO _mysql_db.{dest.table_name} SELECT * FROM _frame")
                case _:
                    raise NotImplementedError(f"DuckDB writer does not support {dest.database_type}")
        else:
            raise ValueError(f"Unknown destination type: {type(dest)}")

