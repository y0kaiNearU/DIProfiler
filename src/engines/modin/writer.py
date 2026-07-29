from __future__ import annotations

import narwhals as nw

from engine_selection.writer import Writer
from engines.modin import CAPABILITIES
from engines.modin.file import writer as file_writer
from models.models import EngineType, FileSource, PipelineRequest


class ModinWriter(Writer):
    capabilities = CAPABILITIES

    @property
    def engine(self) -> EngineType:
        return EngineType.MODIN

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        assert request.destination is not None
        dest = request.destination.source
        if isinstance(dest, FileSource):
            return file_writer.write(frame, dest)
        raise ValueError(f"Modin writer does not support destination type: {type(dest)}")
