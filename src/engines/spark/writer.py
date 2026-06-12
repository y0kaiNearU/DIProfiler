from __future__ import annotations

import narwhals as nw

from core.writer import Writer
from engines.spark.base import SparkBase
from engines.spark.database import writer as db_writer
from engines.spark.file import writer as file_writer
from models.models import DatabaseSource, EngineType, FileSource, PipelineRequest


class SparkWriter(SparkBase, Writer):

    @property
    def engine(self) -> EngineType:
        return EngineType.SPARK

    def can_write(self, request: PipelineRequest) -> bool:
        if request.destination is None:
            return False
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return dest.format in file_writer.SUPPORTED_FORMATS
        if isinstance(dest, DatabaseSource):
            return dest.database_type in db_writer.SUPPORTED_DATABASES
        return False

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return file_writer.write(frame, dest)
        if isinstance(dest, DatabaseSource):
            return db_writer.write(frame, dest)
        raise ValueError(f"Unknown destination type: {type(dest)}")
