from __future__ import annotations

import narwhals as nw

from engine_selection.loader import Loader
from engines.spark import CAPABILITIES
from engines.spark.base import SparkBase
from engines.spark.database import loader as db_loader
from engines.spark.file import loader as file_loader
from models.models import DatabaseSource, EngineType, FileSource, PipelineRequest


class SparkLoader(SparkBase, Loader):
    capabilities = CAPABILITIES

    @property
    def engine(self) -> EngineType:
        return EngineType.SPARK

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(self._get_session(), src)
        if isinstance(src, DatabaseSource):
            return db_loader.load(self._get_session(), src)
        raise ValueError(f"Unknown source type: {type(src)}")
