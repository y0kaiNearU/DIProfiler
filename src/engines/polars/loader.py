from __future__ import annotations

import narwhals as nw

from engine_selection.loader import Loader
from engines.polars import CAPABILITIES
from engines.polars.file import loader as file_loader
from models.models import EngineType, FileSource, PipelineRequest


class PolarsLoader(Loader):
    capabilities = CAPABILITIES

    @property
    def engine(self) -> EngineType:
        return EngineType.POLARS

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(src)
        raise ValueError(f"Polars loader does not support source type: {type(src)}")
