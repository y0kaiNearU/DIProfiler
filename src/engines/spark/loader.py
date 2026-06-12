from __future__ import annotations

import narwhals as nw

from core.loader import Loader
from models.models import DatabaseSource, EngineType, FileFormat, FileSource, PipelineRequest


class SparkLoader(Loader):

    @property
    def engine(self) -> EngineType:
        return EngineType.SPARK

    def can_load(self, request: PipelineRequest) -> bool:
        src = request.source.source
        if isinstance(src, FileSource):
            return src.format in (
                FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON,
                FileFormat.ORC, FileFormat.DELTA,
            )
        elif isinstance(src, DatabaseSource):
            return src.database_type == "postgresql"
        return False

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        try:
            from pyspark.sql import SparkSession
        except ImportError as e:
            raise ImportError("PySpark is required: uv add 'diprofiler[spark]'") from e

        spark = SparkSession.builder.getOrCreate()
        src = request.source.source

        if isinstance(src, FileSource):
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
                case _:
                    raise NotImplementedError(f"Spark loader does not support {src.format}")
        elif isinstance(src, DatabaseSource):
            if src.database_type in ("postgresql", "mysql", "oracle"):
                native = spark.read.format("jdbc").option("url", src.connection_string).option(
                    "dbtable", src.table_name
                ).load()
            else:
                raise NotImplementedError(f"Spark loader does not support {src.database_type}")
        else:
            raise ValueError(f"Unknown source type: {type(src)}")

        return nw.from_native(native)

