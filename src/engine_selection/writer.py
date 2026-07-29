from abc import ABC, abstractmethod
from typing import ClassVar

import narwhals as nw

from engine_selection.capabilities import Capability
from engine_selection.capability_config import required_write_capabilities
from models.models import EngineType, PipelineRequest


class Writer(ABC):
    """
    can_write is decided by matching required_write_capabilities(request.destination)
    against `capabilities`, which each concrete writer sets to its engine's
    CAPABILITIES list (see engines/<engine>/__init__.py) — the same list used
    for engine recommendation, so support is declared in exactly one place.
    """

    capabilities: ClassVar[list[Capability]] = []

    @property
    @abstractmethod
    def engine(self) -> EngineType: ...

    def can_write(self, request: PipelineRequest) -> bool:
        if request.destination is None:
            return False
        required = required_write_capabilities(request.destination)
        return all(any(cap.matches(req) for cap in self.capabilities) for req in required)

    @abstractmethod
    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None: ...
