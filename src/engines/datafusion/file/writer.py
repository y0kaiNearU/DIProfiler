from __future__ import annotations

import os

import narwhals as nw
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.json as pajson
import pyarrow.parquet as pq

from models.models import FileFormat, FileSource, WriteMode

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def write(frame: nw.LazyFrame, dest: FileSource) -> None:
    table = nw.to_native(frame.collect()).to_arrow()

    append = dest.write_mode == WriteMode.APPEND and os.path.exists(dest.path)

    try:
        match dest.format:
            case FileFormat.PARQUET:
                if append:
                    table = pa.concat_tables([pq.read_table(dest.path), table])
                pq.write_table(table, dest.path)
            case FileFormat.CSV:
                if append:
                    existing = pacsv.read_csv(dest.path)
                    table = pa.concat_tables([existing, table])
                with pacsv.CSVWriter(dest.path, table.schema) as writer:
                    writer.write_table(table)
            case FileFormat.JSON:
                if append:
                    existing = pajson.read_json(dest.path)
                    table = pa.concat_tables([existing, table])
                _write_ndjson(table, dest.path)
            case _:
                raise NotImplementedError(f"DataFusion file writer does not support {dest.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"DataFusion failed to write {dest.format.value} to '{dest.path}': {e}") from e


def _write_ndjson(table: pa.Table, path: str) -> None:
    import json
    with open(path, "w") as f:
        for batch in table.to_batches():
            for row in batch.to_pylist():
                f.write(json.dumps(row) + "\n")
