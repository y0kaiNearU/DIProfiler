from __future__ import annotations

import os

import narwhals as nw

from models.models import FileFormat, FileSource, WriteMode


def write(frame: nw.LazyFrame, dest: FileSource) -> None:
    try:
        import modin.pandas as pd
    except ImportError as e:
        raise ImportError('Modin is required: uv add "modin[dask]"') from e

    df = frame.collect().to_pandas()
    df = pd.DataFrame(df)
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
            case _:
                raise NotImplementedError(f"Modin file writer does not support {dest.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Modin failed to write {dest.format.value} to '{dest.path}': {e}") from e
