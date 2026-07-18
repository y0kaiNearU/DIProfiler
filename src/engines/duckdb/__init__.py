from __future__ import annotations

from engine_selection.capabilities import SupportsDataSource, SupportsFormat
from models.models import FileFormat

CAPABILITIES = [
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
    SupportsDataSource("sqlite", "read"),
    SupportsDataSource("sqlite", "write"),
]
