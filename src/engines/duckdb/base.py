from __future__ import annotations

from typing import Any, Callable


class DuckDBBase:

    def __init__(self, factory: Callable[[], Any] | None = None) -> None:
        self._factory = factory
        self._connection = None

    def _get_connection(self) -> Any:
        if self._connection is None:
            if self._factory is not None:
                self._connection = self._factory()
            else:
                try:
                    import duckdb
                except ImportError as e:
                    raise ImportError("DuckDB is required: uv add duckdb") from e
                self._connection = duckdb.connect()
        return self._connection
