from __future__ import annotations

from engine_selection.capabilities import (
    Capability,
    CapabilityRegistry,
    SupportsDataSource,
    SupportsFormat,
)
from engines.arrow import CAPABILITIES as ARROW_CAPABILITIES
from engines.dask import CAPABILITIES as DASK_CAPABILITIES
from engines.duckdb import CAPABILITIES as DUCKDB_CAPABILITIES
from engines.modin import CAPABILITIES as MODIN_CAPABILITIES
from engines.pandas import CAPABILITIES as PANDAS_CAPABILITIES
from engines.polars import CAPABILITIES as POLARS_CAPABILITIES
from engines.spark import CAPABILITIES as SPARK_CAPABILITIES
from models.models import DatasetInfo, EngineType, FileSource, PipelineRequest

_ENGINE_CAPABILITIES = {
    EngineType.DUCKDB: DUCKDB_CAPABILITIES,
    EngineType.SPARK: SPARK_CAPABILITIES,
    EngineType.POLARS: POLARS_CAPABILITIES,
    EngineType.DASK: DASK_CAPABILITIES,
    EngineType.PANDAS: PANDAS_CAPABILITIES,
    EngineType.ARROW: ARROW_CAPABILITIES,
    EngineType.MODIN: MODIN_CAPABILITIES,
}


def build_default_capabilities() -> CapabilityRegistry:
    """Build default capability registry from each engine's declared capabilities."""
    registry = CapabilityRegistry()
    for engine, capabilities in _ENGINE_CAPABILITIES.items():
        registry.register(engine, *capabilities)
    return registry


def _required_capabilities(info: DatasetInfo, direction: str) -> list[Capability]:
    """Capabilities needed to read/write info.source, per direction ("read" or "write")."""
    source = info.source
    if isinstance(source, FileSource):
        return [
            SupportsFormat(source.format, direction),
            SupportsDataSource("filesystem", direction),
        ]
    return [SupportsDataSource(source.database_type, direction)]


def required_read_capabilities(info: DatasetInfo) -> list[Capability]:
    """Capabilities an engine needs to read this dataset as a source."""
    return _required_capabilities(info, "read")


def required_write_capabilities(info: DatasetInfo) -> list[Capability]:
    """Capabilities an engine needs to write this dataset as a destination."""
    return _required_capabilities(info, "write")


def build_required_capabilities(request: PipelineRequest) -> list[Capability]:
    """Build list of capabilities required by request."""
    caps = required_read_capabilities(request.source)
    if request.destination is not None:
        caps += required_write_capabilities(request.destination)
    return caps

