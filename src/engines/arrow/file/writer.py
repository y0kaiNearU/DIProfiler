from __future__ import annotations

import json
import os

import narwhals as nw

from models.models import FileFormat, FileSource, WriteMode

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def write(frame: nw.LazyFrame, dest: FileSource) -> None:
    try:
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        import pyarrow.json as pa_json
        import pyarrow.parquet as pa_parquet
    except ImportError as e:
        raise ImportError("PyArrow is required: uv add pyarrow") from e

    # narwhals' own to_arrow() works regardless of which engine produced the
    # frame (pandas, etc.), no native-object assumptions needed.
    table = frame.collect().to_arrow()
    append = dest.write_mode == WriteMode.APPEND and os.path.exists(dest.path)

    try:
        match dest.format:
            case FileFormat.PARQUET:
                if append:
                    table = pa.concat_tables([pa_parquet.read_table(dest.path), table])
                pa_parquet.write_table(table, dest.path)
            case FileFormat.CSV:
                if append:
                    table = pa.concat_tables([pa_csv.read_csv(dest.path), table])
                pa_csv.write_csv(table, dest.path)
            case FileFormat.JSON:
                if append:
                    table = pa.concat_tables([pa_json.read_json(dest.path), table])
                # PyArrow has no JSON writer; serialize the rows directly.
                with open(dest.path, "w") as f:
                    f.write("\n".join(json.dumps(row, default=str) for row in table.to_pylist()))
            case _:
                raise NotImplementedError(f"Arrow file writer does not support {dest.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"Arrow failed to write {dest.format.value} to '{dest.path}': {e}") from e
