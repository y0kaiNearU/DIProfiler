from __future__ import annotations

import narwhals as nw

from models.models import FileFormat, FileSource

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def load(src: FileSource) -> nw.LazyFrame:
    try:
        import polars as pl
    except ImportError as e:
        raise ImportError("Polars is required: uv add polars") from e

    try:
        match src.format:
            case FileFormat.CSV:
                native = pl.scan_csv(src.path)
            case FileFormat.PARQUET:
                native = pl.scan_parquet(src.path)
            case FileFormat.JSON:
                native = pl.scan_ndjson(src.path)
            case _:
                raise NotImplementedError(f"Polars file loader does not support {src.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Polars failed to read {src.format.value} from '{src.path}': {e}") from e

    return nw.from_native(native)
