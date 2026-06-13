from __future__ import annotations

from core.capabilities import (
    CapabilityRegistry,
    SupportsDataSource,
    SupportsFormat,
)
from models.models import EngineType, FileFormat, PipelineRequest, DatabaseSource, FileSource


def build_default_capabilities() -> CapabilityRegistry:
    """Build default capability registry for DuckDB and Spark."""
    registry = CapabilityRegistry()

    # DuckDB capabilities
    registry.register(
        EngineType.DUCKDB,
        SupportsFormat(FileFormat.CSV, "read"),
        SupportsFormat(FileFormat.PARQUET, "read"),
        SupportsFormat(FileFormat.JSON, "read"),
        SupportsFormat(FileFormat.CSV, "write"),
        SupportsFormat(FileFormat.PARQUET, "write"),
        SupportsFormat(FileFormat.JSON, "write"),
        SupportsDataSource("filesystem", "read"),
        SupportsDataSource("filesystem", "write"),
        SupportsDataSource("postgresql", "read"),
        SupportsDataSource("postgresql", "write"),
        SupportsDataSource("mysql", "read"),
        SupportsDataSource("mysql", "write"),
    )

    # Spark capabilities
    registry.register(
        EngineType.SPARK,
        SupportsFormat(FileFormat.CSV, "read"),
        SupportsFormat(FileFormat.PARQUET, "read"),
        SupportsFormat(FileFormat.JSON, "read"),
        SupportsFormat(FileFormat.ORC, "read"),
        SupportsFormat(FileFormat.DELTA, "read"),
        SupportsFormat(FileFormat.ICEBERG, "read"),
        SupportsFormat(FileFormat.CSV, "write"),
        SupportsFormat(FileFormat.PARQUET, "write"),
        SupportsFormat(FileFormat.JSON, "write"),
        SupportsFormat(FileFormat.ORC, "write"),
        SupportsFormat(FileFormat.DELTA, "write"),
        SupportsFormat(FileFormat.ICEBERG, "write"),
        SupportsDataSource("filesystem", "read"),
        SupportsDataSource("filesystem", "write"),
        SupportsDataSource("postgresql", "read"),
        SupportsDataSource("postgresql", "write"),
        SupportsDataSource("mysql", "read"),
        SupportsDataSource("mysql", "write"),
        SupportsDataSource("oracle", "read"),
        SupportsDataSource("oracle", "write"),
    )

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

