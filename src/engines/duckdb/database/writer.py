from __future__ import annotations

from typing import Any

import narwhals as nw

from engines.duckdb.database.common import attach, qualified_table, setup_extension
from models.models import DatabaseSource

SUPPORTED_DATABASES = ("postgresql", "mysql")


def write(conn: Any, frame: nw.LazyFrame, dest: DatabaseSource) -> None:
    setup_extension(conn, dest.database_type)
    attach(conn, dest)
    conn.register("_frame", nw.to_native(frame.collect()))

    target = qualified_table(dest)
    try:
        conn.execute(f"INSERT INTO {target} SELECT * FROM _frame")
    except Exception as e:
        raise RuntimeError(
            f"DuckDB failed to write to {dest.database_type} "
            f"'{dest.schema}.{dest.table_name}': {e}"
        ) from e
