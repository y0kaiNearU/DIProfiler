from __future__ import annotations

import os

import narwhals as nw

from models.models import FileFormat, FileSource, WriteMode

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def write(frame: nw.LazyFrame, dest: FileSource) -> None:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("Pandas is required: uv add pandas") from e

    df = frame.collect().to_pandas()
    append = dest.write_mode == WriteMode.APPEND and os.path.exists(dest.path)

    try:
        match dest.format:
            case FileFormat.PARQUET:
                if append:
                    df = pd.concat([pd.read_parquet(dest.path), df], ignore_index=True)
                df.to_parquet(dest.path, index=False)
            case FileFormat.CSV:
                if append:
                    df = pd.concat([pd.read_csv(dest.path), df], ignore_index=True)
                df.to_csv(dest.path, index=False)
            case FileFormat.JSON:
                if append:
                    df = pd.concat([pd.read_json(dest.path, lines=True), df], ignore_index=True)
                df.to_json(dest.path, orient="records", lines=True)
            case _:
                raise NotImplementedError(f"Pandas file writer does not support {dest.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Pandas failed to write {dest.format.value} to '{dest.path}': {e}") from e
