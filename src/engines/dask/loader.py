from __future__ import annotations

import narwhals as nw

from engine_selection.loader import Loader
from engines.dask import CAPABILITIES
from engines.dask.base import DaskBase
from engines.dask.file import loader as file_loader
from models.models import EngineType, FileSource, PipelineRequest


class DaskLoader(DaskBase, Loader):
    capabilities = CAPABILITIES

    @property
    def engine(self) -> EngineType:
        return EngineType.DASK

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        self._get_client()
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(src)
        raise ValueError(f"Dask loader does not support source type: {type(src)}")
