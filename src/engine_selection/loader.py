from abc import ABC, abstractmethod
from typing import ClassVar

import narwhals as nw

from engine_selection.capabilities import Capability
from engine_selection.capability_config import required_read_capabilities
from models.models import EngineType, PipelineRequest


class Loader(ABC):
    """
    can_load is decided by matching required_read_capabilities(request.source)
    against `capabilities`, which each concrete loader sets to its engine's
    CAPABILITIES list (see engines/<engine>/__init__.py) — the same list used
    for engine recommendation, so support is declared in exactly one place.
    """

    capabilities: ClassVar[list[Capability]] = []

    @property
    @abstractmethod
    def engine(self) -> EngineType: ...

    def can_load(self, request: PipelineRequest) -> bool:
        required = required_read_capabilities(request.source)
        return all(any(cap.matches(req) for cap in self.capabilities) for req in required)

    @abstractmethod
    def load(self, request: PipelineRequest) -> nw.LazyFrame: ...
