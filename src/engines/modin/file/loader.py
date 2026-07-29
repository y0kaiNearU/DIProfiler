from __future__ import annotations

import narwhals as nw

from models.models import FileFormat, FileSource


def load(src: FileSource) -> nw.LazyFrame:
    try:
        import modin.pandas as pd
    except ImportError as e:
        raise ImportError('Modin is required: uv add "modin[dask]"') from e

    try:
        match src.format:
            case FileFormat.CSV:
                native = pd.read_csv(src.path)
            case FileFormat.PARQUET:
                native = pd.read_parquet(src.path)
            case _:
                raise NotImplementedError(f"Modin file loader does not support {src.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Modin failed to read {src.format.value} from '{src.path}': {e}") from e

    lazy_frame: nw.LazyFrame = nw.from_native(native).lazy()
    return lazy_frame
