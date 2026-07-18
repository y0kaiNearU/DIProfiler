from __future__ import annotations

from core.capabilities import CapabilityRegistry, SupportsDataSource, SupportsFormat
from engines.dask import CAPABILITIES as DASK_CAPABILITIES
from engines.datafusion import CAPABILITIES as DATAFUSION_CAPABILITIES
from engines.duckdb import CAPABILITIES as DUCKDB_CAPABILITIES
from engines.polars import CAPABILITIES as POLARS_CAPABILITIES
from engines.spark import CAPABILITIES as SPARK_CAPABILITIES
from models.models import EngineType, PipelineRequest, DatabaseSource, FileSource

_ENGINE_CAPABILITIES = {
    EngineType.DUCKDB: DUCKDB_CAPABILITIES,
    EngineType.DATAFUSION: DATAFUSION_CAPABILITIES,
    EngineType.SPARK: SPARK_CAPABILITIES,
    EngineType.POLARS: POLARS_CAPABILITIES,
    EngineType.DASK: DASK_CAPABILITIES,
}


def build_default_capabilities() -> CapabilityRegistry:
    """Build default capability registry from each engine's declared capabilities."""
    registry = CapabilityRegistry()
    for engine, capabilities in _ENGINE_CAPABILITIES.items():
        registry.register(engine, *capabilities)
    return registry


def build_required_capabilities(request: PipelineRequest) -> list:
    """Build list of capabilities required by request."""
    caps = []

    # Source capabilities
    if isinstance(request.source.source, FileSource):
        caps.append(SupportsFormat(request.source.source.format, "read"))
        caps.append(SupportsDataSource("filesystem", "read"))
    elif isinstance(request.source.source, DatabaseSource):
        caps.append(SupportsDataSource(request.source.source.database_type, "read"))

    # Destination capabilities
    if request.destination is not None:
        if isinstance(request.destination.source, FileSource):
            caps.append(SupportsFormat(request.destination.source.format, "write"))
            caps.append(SupportsDataSource("filesystem", "write"))
        elif isinstance(request.destination.source, DatabaseSource):
            caps.append(SupportsDataSource(request.destination.source.database_type, "write"))

    return caps

