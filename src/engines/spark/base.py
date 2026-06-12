from __future__ import annotations

from typing import Any, Callable


class SparkBase:

    def __init__(self, factory: Callable[[], Any] | None = None) -> None:
        self._factory = factory
        self._session = None

    def _get_session(self) -> Any:
        if self._session is None:
            if self._factory is not None:
                self._session = self._factory()
            else:
                try:
                    from pyspark.sql import SparkSession
                except ImportError as e:
                    raise ImportError("PySpark is required: uv add 'diprofiler[spark]'") from e
                self._session = SparkSession.builder.getOrCreate()
        return self._session
