from __future__ import annotations

import narwhals as nw

from engine_selection.loader import Loader
from engines.modin import CAPABILITIES
from engines.modin.file import loader as file_loader
from models.models import EngineType, FileSource, PipelineRequest


class ModinLoader(Loader):
    capabilities = CAPABILITIES

    @property
    def engine(self) -> EngineType:
        return EngineType.MODIN

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(src)
        raise ValueError(f"Modin loader does not support source type: {type(src)}")
