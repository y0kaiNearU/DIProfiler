from __future__ import annotations

import narwhals as nw

from models.models import FileFormat, FileSource

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def load(src: FileSource) -> nw.LazyFrame:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("Pandas is required: uv add pandas") from e

    try:
        match src.format:
            case FileFormat.CSV:
                native = pd.read_csv(src.path)
            case FileFormat.PARQUET:
                native = pd.read_parquet(src.path)
            case FileFormat.JSON:
                native = pd.read_json(src.path, lines=True)
            case _:
                raise NotImplementedError(f"Pandas file loader does not support {src.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Pandas failed to read {src.format.value} from '{src.path}': {e}") from e

    lazy_frame: nw.LazyFrame = nw.from_native(native).lazy()
    return lazy_frame
