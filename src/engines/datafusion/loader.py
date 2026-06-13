from __future__ import annotations

import narwhals as nw

from core.loader import Loader
from engines.datafusion.base import DataFusionBase
from engines.datafusion.file import loader as file_loader
from models.models import EngineType, FileSource, PipelineRequest


class DataFusionLoader(DataFusionBase, Loader):

    @property
    def engine(self) -> EngineType:
        return EngineType.DATAFUSION

    def can_load(self, request: PipelineRequest) -> bool:
        src = request.source.source
        if isinstance(src, FileSource):
            return src.format in file_loader.SUPPORTED_FORMATS
        return False

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(self._get_context(), src)
        raise ValueError(f"DataFusion loader does not support source type: {type(src)}")
