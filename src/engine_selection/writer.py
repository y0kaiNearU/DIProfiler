from abc import ABC, abstractmethod

import narwhals as nw

from models.models import EngineType, PipelineRequest


class Writer(ABC):

    @property
    @abstractmethod
    def engine(self) -> EngineType: ...

    @abstractmethod
    def can_write(self, request: PipelineRequest) -> bool: ...

    @abstractmethod
    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None: ...
