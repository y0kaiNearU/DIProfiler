from __future__ import annotations

from core.capabilities import SupportsDataSource, SupportsFormat
from models.models import FileFormat

CAPABILITIES = [
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
]
