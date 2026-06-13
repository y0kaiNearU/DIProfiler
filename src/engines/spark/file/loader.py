from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import FileFormat, FileSource

SUPPORTED_FORMATS = (
    FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON,
    FileFormat.ORC, FileFormat.DELTA, FileFormat.ICEBERG,
)


def load(spark: Any, src: FileSource) -> nw.LazyFrame:
    match src.format:
        case FileFormat.CSV:
            native = spark.read.option("header", "true").option("inferSchema", "true").csv(src.path)
        case FileFormat.PARQUET:
            native = spark.read.parquet(src.path)
        case FileFormat.JSON:
            native = spark.read.json(src.path)
        case FileFormat.ORC:
            native = spark.read.orc(src.path)
        case FileFormat.DELTA:
            native = spark.read.format("delta").load(src.path)
        case FileFormat.ICEBERG:
            native = spark.read.format("iceberg").load(src.path)
        case _:
            raise NotImplementedError(f"Spark file loader does not support {src.format}")
    return nw.from_native(native)
