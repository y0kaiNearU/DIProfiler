from __future__ import annotations

import os

import narwhals as nw

from models.models import FileFormat, FileSource


def _json_read_path(path: str) -> str:
    """Dask writes ndjson as multiple part-files under a directory; glob them if so."""
    return os.path.join(path, "*") if os.path.isdir(path) else path


def load(src: FileSource) -> nw.LazyFrame:
    try:
        import dask.dataframe as dd
    except ImportError as e:
        raise ImportError('Dask is required: uv add "dask[dataframe,distributed]"') from e

    try:
        match src.format:
            case FileFormat.CSV:
                native = dd.read_csv(src.path)
            case FileFormat.PARQUET:
                native = dd.read_parquet(src.path)
            case FileFormat.JSON:
                native = dd.read_json(_json_read_path(src.path), lines=True)
            case _:
                raise NotImplementedError(f"Dask file loader does not support {src.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Dask failed to read {src.format.value} from '{src.path}': {e}") from e

    frame: nw.LazyFrame = nw.from_native(native)
    return frame
