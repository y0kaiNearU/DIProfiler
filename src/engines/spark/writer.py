from __future__ import annotations

import narwhals as nw

from core.writer import Writer
from models.models import DatabaseSource, EngineType, FileFormat, FileSource, PipelineRequest


class SparkWriter(Writer):

    @property
    def engine(self) -> EngineType:
        return EngineType.SPARK

    def can_write(self, request: PipelineRequest) -> bool:
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return dest.format in (
                FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON,
                FileFormat.ORC, FileFormat.DELTA,
            )
        elif isinstance(dest, DatabaseSource):
            return dest.database_type == "postgresql"
        return False

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        try:
            pass  # pyspark is imported transitively via the native frame
        except ImportError as e:
            raise ImportError("PySpark is required: uv add 'diprofiler[spark]'") from e

        dest = request.destination.source
        df = nw.to_native(frame)

        if isinstance(dest, FileSource):
            writer = df.write.mode("overwrite")
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
                case _:
                    raise NotImplementedError(f"Spark writer does not support {dest.format}")
        elif isinstance(dest, DatabaseSource):
            if dest.database_type == "postgresql":
                df.write.format("jdbc").option("url", dest.connection_string).option(
                    "dbtable", dest.table_name
                ).option("user", "").option("password", "").mode("overwrite").save()
            else:
                raise NotImplementedError(f"Spark writer does not support {dest.database_type}")
        else:
            raise ValueError(f"Unknown destination type: {type(dest)}")

