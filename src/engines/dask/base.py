from __future__ import annotations

from typing import Any, Callable


class DaskBase:

    def __init__(self, factory: Callable[[], Any] | None = None) -> None:
        self._factory = factory
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self._factory is not None:
                self._client = self._factory()
            else:
                try:
                    from distributed import Client
                except ImportError as e:
                    raise ImportError('Dask is required: uv add "dask[dataframe,distributed]"') from e
                # processes=False keeps the default cluster in-process (threads only),
                # avoiding surprise subprocess spawning for a library default.
                self._client = Client(processes=False)
        return self._client
