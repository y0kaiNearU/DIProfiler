from __future__ import annotations

import os
import shutil

import narwhals as nw
import pandas as pd

from engines.dask.file.loader import _json_read_path
from models.models import FileFormat, FileSource, WriteMode

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def write(frame: nw.LazyFrame, dest: FileSource) -> None:
    try:
        import dask.dataframe as dd
    except ImportError as e:
        raise ImportError('Dask is required: uv add "dask[dataframe,distributed]"') from e

    pdf = frame.collect().to_pandas()
    append = dest.write_mode == WriteMode.APPEND and os.path.exists(dest.path)

    try:
        if append:
            match dest.format:
                case FileFormat.PARQUET:
                    existing = dd.read_parquet(dest.path).compute()
                case FileFormat.CSV:
                    existing = dd.read_csv(dest.path).compute()
                case FileFormat.JSON:
                    existing = dd.read_json(_json_read_path(dest.path), lines=True).compute()
                case _:
                    raise NotImplementedError(f"Dask file writer does not support append for {dest.format}")
            pdf = pd.concat([existing, pdf], ignore_index=True)

        ddf = dd.from_pandas(pdf, npartitions=1)
        match dest.format:
            case FileFormat.PARQUET:
                if os.path.isdir(dest.path):
                    shutil.rmtree(dest.path)
                ddf.to_parquet(dest.path, write_index=False)
            case FileFormat.CSV:
                ddf.to_csv(dest.path, index=False, single_file=True)
            case FileFormat.JSON:
                if os.path.isdir(dest.path):
                    shutil.rmtree(dest.path)
                ddf.to_json(dest.path)
            case _:
                raise NotImplementedError(f"Dask file writer does not support {dest.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Dask failed to write {dest.format.value} to '{dest.path}': {e}") from e
