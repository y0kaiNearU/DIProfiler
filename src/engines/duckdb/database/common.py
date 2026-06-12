from __future__ import annotations

from typing import Any

from models.models import DatabaseSource

_EXTENSIONS = {
    "postgresql": "postgres_scanner",
    "mysql": "mysql",
}

_ATTACH_TYPES = {
    "postgresql": "postgres",
    "mysql": "mysql",
}


_HAS_SCHEMA = {"postgresql"}


def _qi(name: str) -> str:
    """Quote a SQL identifier to prevent injection."""
    return '"' + name.replace('"', '""') + '"'


def setup_extension(conn: Any, database_type: str) -> None:
    ext = _EXTENSIONS[database_type]
    try:
        conn.install_extension(ext)
        conn.load_extension(ext)
    except Exception as e:
        raise RuntimeError(f"Failed to load DuckDB '{ext}' extension: {e}") from e


def attach(conn: Any, src: DatabaseSource) -> None:
    """Attach the external database as '_db' on the connection."""
    attach_type = _ATTACH_TYPES[src.database_type]
    try:
        conn.execute(
            f"ATTACH IF NOT EXISTS ? AS _db (TYPE {attach_type})",
            [src.connection_string],
        )
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to {src.database_type}: {e}"
        ) from e


def qualified_table(src: DatabaseSource) -> str:
    """Return a quoted, fully-qualified table reference against the '_db' attachment."""
    table = f"_db.{_qi(src.table_name)}"
    if src.database_type in _HAS_SCHEMA:
        table = f"_db.{_qi(src.schema)}.{_qi(src.table_name)}"
    return table
