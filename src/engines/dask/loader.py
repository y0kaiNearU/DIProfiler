from __future__ import annotations

import narwhals as nw

from core.loader import Loader
from engines.dask.base import DaskBase
from engines.dask.file import loader as file_loader
from models.models import EngineType, FileSource, PipelineRequest


class DaskLoader(DaskBase, Loader):

    @property
    def engine(self) -> EngineType:
        return EngineType.DASK

    def can_load(self, request: PipelineRequest) -> bool:
        src = request.source.source
        if isinstance(src, FileSource):
            return src.format in file_loader.SUPPORTED_FORMATS
        return False

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        self._get_client()
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(src)
        raise ValueError(f"Dask loader does not support source type: {type(src)}")
