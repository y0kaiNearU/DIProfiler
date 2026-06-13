from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import FileFormat, FileSource

SUPPORTED_FORMATS = (
    FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON,
    FileFormat.ORC, FileFormat.DELTA, FileFormat.ICEBERG,
)


def write(frame: nw.LazyFrame, dest: FileSource) -> None:
    writer = nw.to_native(frame).write.mode(dest.write_mode.value)
    match dest.format:
        case FileFormat.PARQUET:
            writer.parquet(dest.path)
        case FileFormat.CSV:
            writer.option("header", "true").csv(dest.path)
        case FileFormat.JSON:
            writer.json(dest.path)
        case FileFormat.ORC:
            writer.orc(dest.path)
        case FileFormat.DELTA:
            writer.format("delta").save(dest.path)
        case FileFormat.ICEBERG:
            writer.format("iceberg").save(dest.path)
        case _:
            raise NotImplementedError(f"Spark file writer does not support {dest.format}")
