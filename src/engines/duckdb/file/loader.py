from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import FileFormat, FileSource


def load(conn: Any, src: FileSource) -> nw.LazyFrame:
    frame: nw.LazyFrame
    match src.format:
        case FileFormat.CSV:
            frame = nw.from_native(conn.read_csv(src.path))
        case FileFormat.PARQUET:
            frame = nw.from_native(conn.read_parquet(src.path))
        case FileFormat.JSON:
            frame = nw.from_native(conn.read_json(src.path))
        case _:
            raise NotImplementedError(f"DuckDB file loader does not support {src.format}")
    return frame
