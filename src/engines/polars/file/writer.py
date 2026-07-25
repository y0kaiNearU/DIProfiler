from __future__ import annotations

import os

import narwhals as nw

from models.models import FileFormat, FileSource, WriteMode

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def write(frame: nw.LazyFrame, dest: FileSource) -> None:
    try:
        import polars as pl  # noqa: F401 — imported for the ImportError message below
    except ImportError as e:
        raise ImportError("Polars is required: uv add polars") from e

    # narwhals' own to_polars() works regardless of which engine produced the
    # frame (pandas, etc.), no arrow round-trip needed.
    df = frame.collect().to_polars()
    append = dest.write_mode == WriteMode.APPEND and os.path.exists(dest.path)

    try:
        match dest.format:
            case FileFormat.PARQUET:
                if append:
                    df = pl.concat([pl.read_parquet(dest.path), df])
                df.write_parquet(dest.path)
            case FileFormat.CSV:
                if append:
                    df = pl.concat([pl.read_csv(dest.path), df])
                df.write_csv(dest.path)
            case FileFormat.JSON:
                if append:
                    df = pl.concat([pl.read_ndjson(dest.path), df])
                df.write_ndjson(dest.path)
            case _:
                raise NotImplementedError(f"Polars file writer does not support {dest.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Polars failed to write {dest.format.value} to '{dest.path}': {e}") from e
