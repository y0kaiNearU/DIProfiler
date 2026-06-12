from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import DatabaseSource

SUPPORTED_DATABASES = ("postgresql", "mysql")


def write(conn: Any, frame: nw.LazyFrame, dest: DatabaseSource) -> None:
    conn.register("_frame", nw.to_native(frame.collect()))
    match dest.database_type:
        case "postgresql":
            conn.install_extension("postgres_scanner")
            conn.load_extension("postgres_scanner")
            conn.execute(f"COPY _frame TO postgres('{dest.connection_string}', '{dest.table_name}')")
        case "mysql":
            conn.install_extension("mysql")
            conn.load_extension("mysql")
            conn.execute(f"ATTACH '{dest.connection_string}' AS _mysql_db (TYPE mysql)")
            conn.execute(f"INSERT INTO _mysql_db.{dest.table_name} SELECT * FROM _frame")
        case _:
            raise NotImplementedError(f"DuckDB database writer does not support {dest.database_type}")
