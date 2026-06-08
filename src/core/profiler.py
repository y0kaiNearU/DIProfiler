from abc import ABC, abstractmethod

from models.models import PipelineRequest, ProfilingResult


class Profiler(ABC):
    """Base interface every profiler must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this profiler."""

    @abstractmethod
    def can_handle(self, request: PipelineRequest) -> bool:
        """Return True if this profiler is applicable to the given request."""

    @abstractmethod
    def profile(self, request: PipelineRequest) -> ProfilingResult:
        """Analyze the request and return a ProfilingResult with recommendations."""
