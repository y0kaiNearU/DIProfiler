from __future__ import annotations

import narwhals as nw

from core.writer import Writer
from models.models import EngineType, FileFormat, PipelineRequest


class SparkWriter(Writer):

    @property
    def engine(self) -> EngineType:
        return EngineType.SPARK

    def can_write(self, request: PipelineRequest) -> bool:
        return (
            request.destination is not None
            and request.destination.format in (
                FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON,
                FileFormat.ORC, FileFormat.DELTA,
            )
        )

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        try:
            pass  # pyspark is imported transitively via the native frame
        except ImportError as e:
            raise ImportError("PySpark is required: uv add 'diprofiler[spark]'") from e

        dest = request.destination
        df = nw.to_native(frame)
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
