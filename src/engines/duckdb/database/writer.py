from __future__ import annotations

from typing import Any

import narwhals as nw

from engines.duckdb.database.common import attach, qualified_table, setup_extension
from models.models import DatabaseSource, WriteMode

SUPPORTED_DATABASES = ("postgresql", "mysql", "sqlite")


def write(conn: Any, frame: nw.LazyFrame, dest: DatabaseSource) -> None:
    setup_extension(conn, dest.database_type)
    attach(conn, dest)
    conn.register("_frame", nw.to_native(frame.collect()))

    target = qualified_table(dest)
    select = dest.query if dest.query else "SELECT * FROM _frame"
    try:
        if dest.write_mode == WriteMode.OVERWRITE:
            conn.execute(f"DELETE FROM {target}")
        conn.execute(f"INSERT INTO {target} {select}")
    except Exception as e:
        location = f"'{dest.schema}.{dest.table_name}'" if not dest.query else "custom query"
        raise RuntimeError(
            f"DuckDB failed to write to {dest.database_type} {location}: {e}"
        ) from e
