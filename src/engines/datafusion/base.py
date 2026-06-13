from __future__ import annotations

from typing import Any, Callable


class DataFusionBase:

    def __init__(self, factory: Callable[[], Any] | None = None) -> None:
        self._factory = factory
        self._ctx = None

    def _get_context(self) -> Any:
        if self._ctx is None:
            if self._factory is not None:
                self._ctx = self._factory()
            else:
                try:
                    import datafusion
                except ImportError as e:
                    raise ImportError("DataFusion is required: uv add datafusion") from e
                self._ctx = datafusion.SessionContext()
        return self._ctx
