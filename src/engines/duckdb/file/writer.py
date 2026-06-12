from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import FileFormat, FileSource

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def write(conn: Any, frame: nw.LazyFrame, dest: FileSource) -> None:
    conn.register("_frame", nw.to_native(frame.collect()))
    match dest.format:
        case FileFormat.PARQUET:
            conn.execute(f"COPY _frame TO '{dest.path}' (FORMAT parquet)")
        case FileFormat.CSV:
            conn.execute(f"COPY _frame TO '{dest.path}' (FORMAT csv, HEADER TRUE)")
        case FileFormat.JSON:
            conn.execute(f"COPY _frame TO '{dest.path}' (FORMAT json)")
        case _:
            raise NotImplementedError(f"DuckDB file writer does not support {dest.format}")
