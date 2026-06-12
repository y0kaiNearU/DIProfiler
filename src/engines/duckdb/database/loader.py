from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import DatabaseSource

SUPPORTED_DATABASES = ("postgresql", "mysql")


def load(conn: Any, src: DatabaseSource) -> nw.LazyFrame:
    match src.database_type:
        case "postgresql":
            conn.install_extension("postgres_scanner")
            conn.load_extension("postgres_scanner")
            native = conn.execute(
                f"SELECT * FROM postgres_scan('{src.connection_string}', '{src.table_name}')"
            ).fetch_arrow_table()
        case "mysql":
            conn.install_extension("mysql")
            conn.load_extension("mysql")
            native = conn.execute(
                f"SELECT * FROM mysql_scan('{src.connection_string}', '{src.table_name}')"
            ).fetch_arrow_table()
        case _:
            raise NotImplementedError(f"DuckDB database loader does not support {src.database_type}")
    return nw.from_native(native)
