from __future__ import annotations

import narwhals as nw

from engine_selection.loader import Loader
from engines.pandas.file import loader as file_loader
from models.models import EngineType, FileSource, PipelineRequest


class PandasLoader(Loader):

    @property
    def engine(self) -> EngineType:
        return EngineType.PANDAS

    def can_load(self, request: PipelineRequest) -> bool:
        src = request.source.source
        if isinstance(src, FileSource):
            return src.format in file_loader.SUPPORTED_FORMATS
        return False

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        src = request.source.source
        if isinstance(src, FileSource):
            return file_loader.load(src)
        raise ValueError(f"Pandas loader does not support source type: {type(src)}")
