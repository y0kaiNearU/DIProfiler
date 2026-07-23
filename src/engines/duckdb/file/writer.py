from __future__ import annotations

import os
from typing import Any

import narwhals as nw

from models.models import FileFormat, FileSource, WriteMode

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)

_FORMAT_READERS = {
    FileFormat.PARQUET: "read_parquet",
    FileFormat.CSV: "read_csv",
    FileFormat.JSON: "read_json",
}


def write(conn: Any, frame: nw.LazyFrame, dest: FileSource) -> None:
    conn.register("_frame", nw.to_native(frame.collect()))

    source = "_frame"
    target_path = dest.path
    appending = dest.write_mode == WriteMode.APPEND and os.path.exists(dest.path)
    if appending:
        reader = _FORMAT_READERS.get(dest.format)
        if reader is None:
            raise NotImplementedError(f"DuckDB file writer does not support append for {dest.format}")
        source = f"(SELECT * FROM {reader}('{dest.path}') UNION ALL SELECT * FROM _frame)"
        # Write to a temp path first: COPYing FROM and TO the same file in one
        # statement corrupts/locks the file (observed on Windows).
        target_path = dest.path + ".tmp"

    match dest.format:
        case FileFormat.PARQUET:
            conn.execute(f"COPY {source} TO '{target_path}' (FORMAT parquet)")
        case FileFormat.CSV:
            conn.execute(f"COPY {source} TO '{target_path}' (FORMAT csv, HEADER TRUE)")
        case FileFormat.JSON:
            conn.execute(f"COPY {source} TO '{target_path}' (FORMAT json)")
        case _:
            raise NotImplementedError(f"DuckDB file writer does not support {dest.format}")

    if appending:
        os.replace(target_path, dest.path)
