from __future__ import annotations

import narwhals as nw

from engine_selection.loader import Loader
from engines.pandas import CAPABILITIES
from engines.pandas.file import loader as file_loader
from models.models import EngineType, FileSource, PipelineRequest


class PandasLoader(Loader):
    capabilities = CAPABILITIES

    @property
    def engine(self) -> EngineType:
        return EngineType.PANDAS

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(src)
        raise ValueError(f"Pandas loader does not support source type: {type(src)}")
