from __future__ import annotations

from typing import Any

import narwhals as nw

from engines.duckdb.database.common import attach, qualified_table, setup_extension
from models.models import DatabaseSource

SUPPORTED_DATABASES = ("postgresql", "mysql", "sqlite")


def load(conn: Any, src: DatabaseSource) -> nw.LazyFrame:
    setup_extension(conn, src.database_type)
    attach(conn, src)

    sql = src.query if src.query else f"SELECT * FROM {qualified_table(src)}"

    try:
        native = conn.execute(sql).fetch_arrow_table()
    except Exception as e:
        location = f"'{src.schema}.{src.table_name}'" if not src.query else "custom query"
        raise RuntimeError(
            f"DuckDB failed to read from {src.database_type} {location}: {e}"
        ) from e

    return nw.from_native(native)
