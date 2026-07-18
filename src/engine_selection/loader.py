from abc import ABC, abstractmethod

import narwhals as nw

from models.models import EngineType, PipelineRequest


class Loader(ABC):

    @property
    @abstractmethod
    def engine(self) -> EngineType: ...

    @abstractmethod
    def can_load(self, request: PipelineRequest) -> bool: ...

    @abstractmethod
    def load(self, request: PipelineRequest) -> nw.LazyFrame: ...
