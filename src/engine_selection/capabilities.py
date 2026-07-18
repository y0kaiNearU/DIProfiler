from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from packaging.version import Version

from models.models import EngineType, FileFormat


def _meets_min_version(min_version: str | None, engine_version: str | None) -> bool:
    """A capability with no min_version, or an unknown engine_version, is always available."""
    if min_version is None or engine_version is None:
        return True
    return Version(engine_version) >= Version(min_version)


class Capability(ABC):
    """Base class for engine capabilities."""

    @abstractmethod
    def matches(self, other: Capability) -> bool:
        """Check if this capability matches another capability requirement."""
        raise NotImplementedError()

    def is_available(self, engine_version: str | None) -> bool:
        """Whether this capability applies at the given engine version. Defaults to always available."""
        return True


@dataclass
class SupportsFormat(Capability):
    """Engine can read or write a specific file format."""

    format: FileFormat
    direction: str  # "read" or "write"
    min_version: str | None = None  # earliest engine version that supports this, if version-gated

    def matches(self, other: Capability) -> bool:
        if not isinstance(other, SupportsFormat):
            return False
        return self.format == other.format and self.direction == other.direction

    def is_available(self, engine_version: str | None) -> bool:
        return _meets_min_version(self.min_version, engine_version)

    def __hash__(self) -> int:
        return hash((self.format, self.direction, self.min_version))


@dataclass
class SupportsDataSource(Capability):
    """Engine can read from or write to a specific data source type."""

    source_type: str  # "filesystem", "s3", "database", "hdfs", etc.
    direction: str  # "read" or "write"
    min_version: str | None = None  # earliest engine version that supports this, if version-gated

    def matches(self, other: Capability) -> bool:
        if not isinstance(other, SupportsDataSource):
            return False
        return self.source_type == other.source_type and self.direction == other.direction

    def is_available(self, engine_version: str | None) -> bool:
        return _meets_min_version(self.min_version, engine_version)

    def __hash__(self) -> int:
        return hash((self.source_type, self.direction, self.min_version))


@dataclass
class SupportsOperations(Capability):
    """Engine supports a list of operations."""

    operations: frozenset[str]

    def __init__(self, operations: list[str] | frozenset[str]):
        if isinstance(operations, list):
            self.operations = frozenset(operations)
        else:
            self.operations = operations

    def matches(self, other: Capability) -> bool:
        if not isinstance(other, SupportsOperations):
            return False
        # Check if all required operations are supported
        return other.operations.issubset(self.operations)

    def __hash__(self) -> int:
        return hash(self.operations)


class CapabilityRegistry:
    """Registry mapping engines to their capabilities."""

    def __init__(self) -> None:
        self._capabilities: dict[EngineType, set[Capability]] = {}

    def register(self, engine: EngineType, *capabilities: Capability) -> None:
        """Register capabilities for an engine."""
        if engine not in self._capabilities:
            self._capabilities[engine] = set()
        self._capabilities[engine].update(capabilities)

    def get(self, engine: EngineType) -> set[Capability]:
        """Get all capabilities for an engine."""
        return self._capabilities.get(engine, set())

    def can_handle(
        self,
        engine: EngineType,
        required: list[Capability],
        engine_version: str | None = None,
    ) -> bool:
        """Check if engine satisfies all required capabilities at the given engine version."""
        if engine not in self._capabilities:
            return False
        engine_caps = self._capabilities[engine]
        return all(
            any(cap.matches(req) and cap.is_available(engine_version) for cap in engine_caps)
            for req in required
        )
