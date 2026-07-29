from __future__ import annotations

import narwhals as nw

from models.models import FileFormat, FileSource


def load(src: FileSource) -> nw.LazyFrame:
    try:
        import pyarrow.csv as pa_csv
        import pyarrow.json as pa_json
        import pyarrow.parquet as pa_parquet
    except ImportError as e:
        raise ImportError("PyArrow is required: uv add pyarrow") from e

    try:
        match src.format:
            case FileFormat.CSV:
                native = pa_csv.read_csv(src.path)
            case FileFormat.PARQUET:
                native = pa_parquet.read_table(src.path)
            case FileFormat.JSON:
                native = pa_json.read_json(src.path)
            case _:
                raise NotImplementedError(f"Arrow file loader does not support {src.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Arrow failed to read {src.format.value} from '{src.path}': {e}") from e

    lazy_frame: nw.LazyFrame = nw.from_native(native).lazy()
    return lazy_frame
