from __future__ import annotations

from core.capabilities import (
    CapabilityRegistry,
    SupportsDataSource,
    SupportsFormat,
)
from models.models import EngineType, FileFormat, PipelineRequest


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
    )

    # Spark capabilities
    registry.register(
        EngineType.SPARK,
        SupportsFormat(FileFormat.CSV, "read"),
        SupportsFormat(FileFormat.PARQUET, "read"),
        SupportsFormat(FileFormat.JSON, "read"),
        SupportsFormat(FileFormat.ORC, "read"),
        SupportsFormat(FileFormat.DELTA, "read"),
        SupportsFormat(FileFormat.CSV, "write"),
        SupportsFormat(FileFormat.PARQUET, "write"),
        SupportsFormat(FileFormat.JSON, "write"),
        SupportsFormat(FileFormat.ORC, "write"),
        SupportsFormat(FileFormat.DELTA, "write"),
        SupportsDataSource("filesystem", "read"),
        SupportsDataSource("filesystem", "write"),
    )

    return registry


def build_required_capabilities(request: PipelineRequest) -> list:
    """Build list of capabilities required by request."""
    caps = []

    caps.append(SupportsFormat(request.source.format, "read"))

    if request.destination is not None:
        caps.append(SupportsFormat(request.destination.format, "write"))
        
    caps.append(SupportsDataSource("filesystem", "read"))
    if request.destination is not None:
        caps.append(SupportsDataSource("filesystem", "write"))

    return caps
