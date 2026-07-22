from __future__ import annotations

import narwhals as nw

from engine_selection.writer import Writer
from engines.pandas.file import writer as file_writer
from models.models import EngineType, FileSource, PipelineRequest


class PandasWriter(Writer):

    @property
    def engine(self) -> EngineType:
        return EngineType.PANDAS

    def can_write(self, request: PipelineRequest) -> bool:
        if request.destination is None:
            return False
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return dest.format in file_writer.SUPPORTED_FORMATS
        return False

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        assert request.destination is not None
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return file_writer.write(frame, dest)
        raise ValueError(f"Pandas writer does not support destination type: {type(dest)}")
