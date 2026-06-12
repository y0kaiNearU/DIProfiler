from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import FileFormat, FileSource

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def load(conn: Any, src: FileSource) -> nw.LazyFrame:
    match src.format:
        case FileFormat.CSV:
            return nw.from_native(conn.read_csv(src.path))
        case FileFormat.PARQUET:
            return nw.from_native(conn.read_parquet(src.path))
        case FileFormat.JSON:
            return nw.from_native(conn.read_json(src.path))
        case _:
            raise NotImplementedError(f"DuckDB file loader does not support {src.format}")
