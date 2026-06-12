from __future__ import annotations

import narwhals as nw

from core.loader import Loader
from engines.spark.base import SparkBase
from engines.spark.database import loader as db_loader
from engines.spark.file import loader as file_loader
from models.models import DatabaseSource, EngineType, FileSource, PipelineRequest


class SparkLoader(SparkBase, Loader):

    @property
    def engine(self) -> EngineType:
        return EngineType.SPARK

    def can_load(self, request: PipelineRequest) -> bool:
        src = request.source.source
        if isinstance(src, FileSource):
            return src.format in file_loader.SUPPORTED_FORMATS
        if isinstance(src, DatabaseSource):
            return src.database_type in db_loader.SUPPORTED_DATABASES
        return False

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(self._get_session(), src)
        if isinstance(src, DatabaseSource):
            return db_loader.load(self._get_session(), src)
        raise ValueError(f"Unknown source type: {type(src)}")
